import json
import boto3
import base64
import time
import traceback
import mimetypes
import concurrent.futures
import logging

# Initialize AWS clients
rekognition_client = boto3.client('rekognition')
textract_client = boto3.client('textract')
dynamodb_client = boto3.client('dynamodb')
s3_client = boto3.client('s3')

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
DYNAMODB_TABLE = 'ParticipationRecordsProj3'
S3_BUCKET_NAME = 'proj3-group27-bucket'

# Predefined S3 object prefixes for reference images
names_image_prefix = "proj3/proj3-images/names/"
face_images_prefix = "proj3/proj3-images/faces/"

def lambda_handler(event, context):
    logger.info("Lambda triggered with event: %s", json.dumps(event))

    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"message": "CORS preflight successful"})
        }

    try:
        # Support both test and API Gateway calls
        if 'body' in event and isinstance(event['body'], str):
            event = json.loads(event['body'])

        # Validate event data
        validate_event_fields(event)

        # Extract input data
        name, email, class_date, uploaded_image_data, uploaded_image_key = extract_event_data(event)

        # If base64 image data is provided, upload the image to S3
        if uploaded_image_data:
            uploaded_image_key = upload_base64_image_to_s3(uploaded_image_data, name, email, class_date)
            if not uploaded_image_key:
                return format_api_response(generate_response(
                    False, name, email, class_date, [], False, False, [], [], [], "Failed to upload image to S3."
                ))

        # Retrieve images from S3
        uploaded_image = get_s3_image(uploaded_image_key)
        names_image_keys = list_s3_files(names_image_prefix)
        face_images_keys = list_s3_files(face_images_prefix)

        if not uploaded_image or not names_image_keys or not face_images_keys:
            return format_api_response(generate_response(
                False, name, email, class_date, [], False, False, [], [], [], "Failed to retrieve required images from S3."
            ))

        # Extract text from name images
        with concurrent.futures.ThreadPoolExecutor() as executor:
            names_image_details = list(executor.map(lambda key: extract_text_from_image(get_s3_image(key)), names_image_keys))

        # Detect faces in images
        with concurrent.futures.ThreadPoolExecutor() as executor:
            uploaded_image_face_details = detect_faces(uploaded_image)
            face_image_face_details = list(executor.map(lambda key: detect_faces(get_s3_image(key)), face_images_keys))

        if not uploaded_image_face_details or not any(face_image_face_details):
            return format_api_response(generate_response(
                False, name, email, class_date, names_image_details, False, False, uploaded_image_face_details, face_image_face_details, [], "No faces detected."
            ))

        # Compare faces
        face_match_details = [compare_faces(uploaded_image, get_s3_image(face_key)) for face_key in face_images_keys]
        face_match = any([match[0] for match in face_match_details])
        face_similarity_scores = [match[1] for match in face_match_details if match[0]]

        # Name match check (case-insensitive partial)
        name_match = any(name.lower() in str(text).lower() for text in names_image_details)

        participation = name_match or face_match

        # Store in DynamoDB
        success = create_participation_record(
            name, email, class_date, participation, name_match, face_match,
            names_image_details, uploaded_image_key
        )

        return format_api_response(generate_response(
            participation, name, email, class_date, names_image_details, name_match, face_match,
            uploaded_image_face_details, face_image_face_details, face_similarity_scores,
            None if success else "Failed to write to DynamoDB."
        ))

    except Exception as e:
        logger.error("Error: %s", str(e))
        traceback.print_exc()
        return format_api_response(generate_response(
            False, "", "", "", [], False, False, [], [], [], f"Unexpected error: {str(e)}"
        ))


def format_api_response(response_dict):
    return {
        "statusCode": 200 if not response_dict.get("error") else 500,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps(response_dict)
    }


def generate_response(
    participation,
    name,
    email,
    class_date,
    extracted_names,
    name_match,
    face_match,
    uploaded_faces,
    reference_faces,
    similarity_scores=[],
    error=None
):
    return {
        "participation": participation,
        "name": name,
        "email": email,
        "class_date": class_date,
        "extracted_names": extracted_names,
        "name_match": name_match,
        "face_match": face_match,
        "uploaded_faces": uploaded_faces,
        "reference_faces": reference_faces,
        "similarity_scores": similarity_scores,
        "error": error
    } 

def validate_event_fields(event):
    """Ensure required fields exist in the event."""
    required_fields = ['name', 'email', 'class_date']
    for field in required_fields:
        if field not in event:
            raise ValueError(f"Missing required field: {field}")


def extract_event_data(event):
    """Extract and normalize input data from the event."""
    name = event['name'].strip().lower()
    email = event['email']
    class_date = event['class_date']
    uploaded_image_data = event.get('uploaded_image_data', None)
    uploaded_image_key = event.get('uploaded_image_key', None)
    return name, email, class_date, uploaded_image_data, uploaded_image_key


def upload_base64_image_to_s3(image_data, name, email, class_date):
    """Uploads a base64 image to S3 and returns the uploaded image's S3 key."""
    try:
        image_bytes = base64.b64decode(image_data)
        content_type = get_content_type_from_bytes(image_bytes)

        if not content_type:
            raise ValueError("Unable to determine image content type")

        timestamp = str(int(time.time()))
        s3_key = f"proj3/proj3-images/uploads/{class_date}/{name}.jpg"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_bytes,
            ContentType=content_type
        )

        logger.info(f"Uploaded image saved at {s3_key}")
        return s3_key
    except Exception as e:
        logger.error("Error uploading image: %s", str(e))
        return None


def get_content_type_from_bytes(image_bytes):
    """Determines the MIME content type of an image."""
    content_type, _ = mimetypes.guess_type('image.jpg')
    return content_type if content_type else "image/jpeg"


def get_s3_image(s3_key):
    """Fetches image from S3 as bytes."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        logger.error(f"Error fetching image {s3_key}: {str(e)}")
        return None


def list_s3_files(prefix):
    """Lists all files in a given S3 directory (prefix)."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        if 'Contents' not in response:
            return []
        return [content['Key'] for content in response['Contents']]
    except Exception as e:
        logger.error(f"Error listing files in S3 with prefix {prefix}: {str(e)}")
        return []


def extract_text_from_image(image_bytes):
    """Uses Textract to extract text from an image."""
    try:
        response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
        return [item["Text"] for item in response.get("Blocks", []) if item["BlockType"] == "LINE"]
    except Exception as e:
        logger.error("Error extracting text: %s", str(e))
        return []


def detect_faces(image_bytes):
    """Detects faces in an image using Rekognition."""
    try:
        response = rekognition_client.detect_faces(Image={'Bytes': image_bytes}, Attributes=['DEFAULT'])
        return response.get("FaceDetails", [])
    except Exception as e:
        logger.error("Error detecting faces: %s", str(e))
        return []


def compare_faces(source_image, target_image):
    """Compares faces between two images using Rekognition and returns a tuple of (match, similarity_score)."""
    try:
        response = rekognition_client.compare_faces(
            SourceImage={'Bytes': source_image},
            TargetImage={'Bytes': target_image},
            SimilarityThreshold=85
        )
        face_matches = response.get('FaceMatches', [])
        if face_matches:
            similarity_score = face_matches[0]['Similarity']
            return True, similarity_score
        return False, 0.0
    except Exception as e:
        logger.error("Error comparing faces: %s", str(e))
        return False, 0.0


def create_participation_record(name, email, class_date, participation, name_match, face_match, extracted_texts, uploaded_image_key):
    """Stores participation data in DynamoDB."""
    try:
        dynamodb_client.put_item(
            TableName=DYNAMODB_TABLE,
            Item={
                'name': {'S': name},
                'email': {'S': email},
                'class_date': {'S': class_date},
                'participation': {'BOOL': participation},
                'name_match': {'BOOL': name_match},
                'face_match': {'BOOL': face_match},
                'uploaded_image_key': {'S': uploaded_image_key}
            }
        )
        return True
    except Exception as e:
        logger.error("Error writing to DynamoDB: %s", str(e))
        return False

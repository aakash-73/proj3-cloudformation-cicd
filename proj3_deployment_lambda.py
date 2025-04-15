import json
import boto3
import urllib.parse

def lambda_handler(event, context):
    # Initialize the Amplify client
    amplify_client = boto3.client('amplify')

    # Get the S3 bucket and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])

    print(f"New zip file uploaded to S3 bucket: {bucket}, key: {key}")

    # Check if the uploaded file is the specific one we're interested in
    if key != 'proj3/index.zip':  # Adjust the path to your specific file
        print(f"Skipped non-target file: {key}")
        return {
            'statusCode': 200,
            'body': json.dumps('No deployment triggered, incorrect file uploaded.')
        }

    # Define the parameters for the Amplify deployment
    params = {
        'appId':'d3dfbfoc7clprp',  # Replace with your Amplify App ID
        'branchName': 'dev',            # Replace with your branch name
        'sourceUrl': f"s3://{bucket}/{key}"
    }

    try:
        # Start the Amplify deployment
        response = amplify_client.start_deployment(**params)
        print(f"Amplify deployment started: {json.dumps(response)}")
        return {
            'statusCode': 200,
            'body': json.dumps('Deployment triggered successfully!')
        }

    except Exception as e:
        # Handle any errors
        print(f"Error starting Amplify deployment: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }

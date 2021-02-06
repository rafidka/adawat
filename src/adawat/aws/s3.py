import contextlib
import boto3
import logging

log = logging.getLogger(__name__)


@contextlib.contextmanager
def open(bucket: str, key: str, session=None):
    """
    Similar to Python's open() method but works with S3.

    This is a utility method that makes it easy to read S3 object in a more
    Pythonic with using the `with` statement. An example use:

    ```
    # Read the first 10 lines of the file.
    with s3open('test-bucket', 'test-key') as body:
        lines = islice(body.iter_lines(), 10)
    ```

    Keyword arguments:
    bucket -- The S3 bucket containing the object.
    key -- The key of the object to open.

    Returns:
    A stream to the S3 object body.
    """
    if session is None:
        log.debug("No session is provided. Using default session.")
        session = boto3.session.Session()
    s3 = session.client('s3')

    body = None
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response['Body']
        yield body
    finally:
        if body is not None:
            body.close()

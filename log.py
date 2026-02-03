from db import database


async def log_file_upload(
    user_name: str,
    action: str,
    file_name: str,
    extension: str,
    file_size: int,
):
    query = """
        INSERT INTO audit_logs
        (user_name, action, file_name, extension, file_size)
        VALUES (:user_name, :action, :file_name, :extension, :file_size)
    """

    await database.execute(
        query=query,
        values={
            "user_name": user_name,
            "action": action,
            "file_name": file_name,
            "extension": extension,
            "file_size": file_size,
        },
    )

import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGJ-8qUqZV2LLh8K4EQYoiThJf8bNfF9ET6sNkOi_lBICd2WhV1VKW-Q9ciujL7xknis-PfFmFTAYlS3I0Hr14mxc0BLIG3hbfI-DZ7miVD78HVVyhfyD_S-wg1gEoQFqvDQJRVSlMeTfzQwlUOrLi5Axx5Yx5r1iUXRdK0LctRX4kvHfcb7qS-Q3VNOo_CyvXVncwi1qeLdFtZUadvEU_PVMkTsMnXLf9pMsEfSmPwe4qJ0ng9RDwaLbAcT3YFnbU-YjxWbJX90onvpCfW-W_8CJC1_UiQ4q0ixd4eraQZk8WM1cDyr5-xVtgaPN9djm1lM3cdWFeoZ_zYN3ltb72NMPWIY285EngP1amP2PeyEiFH_YOkXXXmgQzxoMS9L__RqwOLtZuJWiBWt7-Ee4sDTBT-OS2x3W3UFyt0aIFRKnpXQ3-LC9T90obf0FKkbO40hQ-MEXGa7KM-UKagElfJaEz0L9krE9IYoXQ7XknKP-EPnbeUnSoJ4iKsUGivG6HEnpcIJpaNY_XlPmzmc_iZC8txm0IZeBitTrYacVMUz7tZplBzKhLhT57fWrMBcgVve3JrNpmpReKuz2QkgEVA2i5pjkHMGtWgnX0r911nUd2FHCKMdX-7goA9qoQEGhMkVga9zYnKExYYohrdONYFYVqFr4m6FpoqH1kmiMO51FzEkjXSeR0-bY5AZLETUQffUysLQUCGeVYtmHqL_00K1Qu5vZR3Id3lxc6oZRKfjXVq6zK8Ce-mmRATa9kaA5X8-ftHUwm4QMEVd0TWDeDFM2P9WGbX_AVtG_tJ5cS3Atok2fr42PcKpibL1AXl9sZDXGOSiLBAELxOUIdU0SzF3ejvGI946_co8YKjOVn2dMJeHa5WXvJ5yVdD00kxA9PgwchmwA9YNPWhYqCit73c7D_VLySak94feYwnNB5tra3DuUCiSaPP2Gev3l5jbndaAIAsUXqfIMv0FVeEXfGlfL1SUQ-2V_vHZ8ljDAzkYokyo93uWh9pSblRDIaM9WW96r-41EidWniJNut6mPCObnV7OnTk0DsVu-4PvxDGjvY_d9HzxIjPbJ7p-OHqo6XsE4ZQD76cbM5lTwFCfojZ5R7Qrj4k8JRHPP3nty6BNDIYcLJjdggYOK8eLBcKhgkJdBjDcI9MFvNqvVjYWRRnfGysG2K9m7jDYTymnUx8uQBU0a77rL1j6vCNUSPT9JceaNEdEgJWyZRMJK30pLCOkG3srYcn2Xvmk7HOfV1AQ5oqEdDhdnTd9_euisRZ5cmQ"
CHUNK_SIZE = 150 * 1024 * 1024  # 150MB - A good default, adjustable based on network stability

def upload_large_file(local_path, dropbox_path):
    """
    Uploads a large file to Dropbox using chunked uploads with a static token.
    """
    if not ACCESS_TOKEN:
        print("Error: Please set the DROPBOX_ACCESS_TOKEN environment variable.")
        sys.exit(1)

    try:
        # Instantiate the Dropbox object with the static access token
        dbx = dropbox.Dropbox(ACCESS_TOKEN)
        dbx.users_get_current_account() # Test the connection
        print("Successfully connected to Dropbox.")
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        sys.exit(1)

    file_size = os.path.getsize(local_path)
    print(f"File size: {file_size / (1024*1024*1024):.2f} GB")

    with open(local_path, 'rb') as f:
        try:
            if file_size <= CHUNK_SIZE:
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
                print(f"Successfully uploaded {local_path} to {dropbox_path}")
            else:
                upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
                cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                           offset=f.tell())
                commit = dropbox.files.CommitInfo(path=dropbox_path, mode=dropbox.files.WriteMode('overwrite'))

                print(f"Started upload session: {cursor.session_id}")

                while f.tell() < file_size:
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        dbx.files_upload_session_finish(f.read(CHUNK_SIZE),
                                                         cursor,
                                                         commit)
                        print("\nFile upload complete.")
                    else:
                        try:
                            dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                            cursor.offset = f.tell()
                            # Progress indicator
                            progress = (f.tell() / file_size) * 100
                            sys.stdout.write(f"\rProgress: {progress:.2f}%")
                            sys.stdout.flush()
                        except dropbox.exceptions.ApiError as e:
                            if e.error.is_incorrect_offset():
                                print(f"\nOffset error. Server has {e.error.get_incorrect_offset().correct_offset} bytes. Retrying from that offset.")
                                f.seek(e.error.get_incorrect_offset().correct_offset)
                                cursor.offset = e.error.get_incorrect_offset().correct_offset
                            else:
                                raise

        except Exception as e:
            print(f"\nAn error occurred during upload: {e}")
            sys.exit(1)

def upload_folder(local_folder, dropbox_folder):
    """
    Recursively uploads all files in a folder to Dropbox.
    """
    if not ACCESS_TOKEN:
        print("Error: Please set the DROPBOX_ACCESS_TOKEN environment variable.")
        sys.exit(1)

    try:
        dbx = dropbox.Dropbox(ACCESS_TOKEN)
        dbx.users_get_current_account()
        print("Successfully connected to Dropbox.")
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        sys.exit(1)

    total_files = sum([len(files) for _, _, files in os.walk(local_folder)])
    uploaded_files = 0
    
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_file_path, local_folder)
            dropbox_file_path = os.path.join(dropbox_folder, relative_path).replace(os.sep, '/')
            
            uploaded_files += 1
            print(f"\n[{uploaded_files}/{total_files}] Uploading: {relative_path}")
            upload_large_file(local_file_path, dropbox_file_path)

    print(f"\nFolder upload complete. Uploaded {uploaded_files} files.")

if __name__ == "__main__":
    '''
    example usage:
    python upload_dropbox.py /userfolder/neurips2025.tar.gz /Paper_pred/mineru_data/neurips2025.tar.gz
    '''
    if len(sys.argv) != 3:
        print("Usage: python upload_script.py <local_path> <dropbox_destination_path>")
        print("       local_path can be a file or a folder")
        sys.exit(1)

    local_path = sys.argv[1]
    dropbox_dest = sys.argv[2]

    if not os.path.exists(local_path):
        print(f"Error: Local path '{local_path}' not found.")
        sys.exit(1)

    if os.path.isdir(local_path):
        print(f"Detected folder: {local_path}")
        upload_folder(local_path, dropbox_dest)
    else:
        print(f"Detected file: {local_path}")
        upload_large_file(local_path, dropbox_dest)
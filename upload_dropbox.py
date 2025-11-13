import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGFb4FWpzFSn2rBK6fgjyeayk2_BplpTgnIg0xYn5F89W2agfI-yx5Ef3NkZ4SecBr5xRYRRtHchWf45FsNRHKdo_p4T7ewEQEGzQcJFAjaTAfbzL2iGVzgjUGFmIj6EpV3WgprNSBlxNAsNLPfNIbns80jhDYyGXjvwmgNyrvmU1JM5FaVg6xOJ5OwnXY7ldvnWpdSmiAbeD16B-jvGhuhCsvLbfSgUSND9ncDRExigNBZg4T0V3s1f3QOajYyz9FbiofnklRZmUac2dSaufWM57fw_3CkSXSsNE7GFICDwjl5o4tzsZTjO_lFNc44v0bCc-OXpwpOWrEmpyjDMz61iWlaMcdaZG81sJNccuQDMl3vTYW-1vSmvWluRRyXfG0d1xZa9L4L4zeXqhsKKD6u1ziwFg1cIMjDJDTNhn5NXbyYftpUNjlQSx48qwCwwWvx34lRQypHrakmUE1RvKzvc17RgqiLfwiYsG6CTh8ibi_NWfWRDn_HW_zki8-PT-zGC5VJweb26_RtN3Kukw16g0PP1dZXqAA8Vocy5UyKJrDDnnHi0iYlDkwHZLtSFZt8lMwLhNn0oEbsXiP9POCGQU8VTbqqY5pXxpJTUAe_3XDk_j-smOdGPPlyNHWzB1TrQEAK-YQxP7fPiBSZwfGqU1Fjzot6LNbssDwinwIPUrs8jDkvc4wkAeypETZDwSF7QKUdN2xJw1RVUGTQJ9FbaNCZlyoyRyXT5N3UrP34DpSWBScyYgeQeuvApCvFV6lygjx-piHxu68D5JFKRCerGK8k7sztXE69CyXM-GxOSp5Td_JUsU3klZqSMnLEnXdJMi7bZ0uDBrhLVrJwqg2DBYrnId7fGY62aksJ3LAAeVh-206sGhPRKW9rysZXSEzrTmM6zB91H5tIBVZBspfAljicsSimHetwBw9LhSTAgA49Ae_wQQ8UxhjnB_4Xh7TkGvJXCGF3tKf9vFnUs7m2f7zsF3OZFSn-CrbxcNA_JByRSlWPtzlDr7yJzQeJMMyqYOk0o5mKlC8I62BG14wvx3moH6Bu6zsaBxTzmngzsdLwhkLdpHn_GNBV_VdZzSBbIy34WPc53L6BChRqHisZxB4yV1ofV6cJ9Ayp9xdZYMjBUdXLpHoBwFwz8mix6D31oDUYBz1-jbXZezAxpo9sR1ig5TLqvy3-JRv4zRYMaTt2okl8vPiPSm3QdI9I8fE2KXV12oAm6qkBS1uxsuWPwUBJlzZWPFRsF5Ar6zv7rbLMLt1A66r2uQkziFK3niNI"
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
    python upload_dropbox.py /data/scratch/mpx602/topcon-1/neurips2024.tar.gz /Paper_pred/neurips2024.tar.gz
    python upload_dropbox.py /mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/mineru_data /Paper_pred/mineru_data
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
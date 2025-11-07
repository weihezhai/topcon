import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGEwrYf1IYkQBlWTAT71AWZrcoM4VMW5VmKkTMRnQ-rD2_XJdCsU-bhgpBx4BHi4km8TUs3JSDtQL3ZOj9PEqHF0eJI_nV5lDvqQmzxH1I5tSO_Kp8giuaZfzv2ao_Sy6LccqBDve3T9CYa1NHmzcwbBm1uiHvhouJPDUdb0iSASiyV_bAbSn1QACedFBcD8Ap1YqTOdpfWWptgDoTrn_SlYHGgNgnXk5myta7Wzb8WF_IdzkMt2TGtBy6o6TYspnENgNW7E4qA_FIiZrgkPQVrj5DoY08xc50L7Uf5__GVXUcb39YovnHwg-41IYG440viywq591e2d859I7aeSLO53NV2EgDsYZf5GgL8w-If5LrrUy9nrnZABJZxTBiv3bIOABGZLKF37jZBBQKkIbSMn3ANhjDGUdvV0o_eE6UfP1CvpJU7fMQcYPMFMJ-R9shu4Yr1xh4yH6MvSvzEQtTCGoHHkV9MBXJL-A0Kt_Dx3w09AXu6OSZ5-QLid8fjcKqrydXvCAtq4DM9i0PpU1-1_RVZ13V2g3zMHOjaBoKAR6lCLZ2PzlwdYUtX85sok8AWsLIlBRrKkGfb0KIIyuUe8lHt6eP56m_tJcT6nbHDy78i5EcC0Wa5yxvFBA-BM2zGWb9GZwNUdx-oPE5dCMfQt7KABiJ3Nm7zcggAeMJe_rrnDzc_V2nz6Bc9tBPW1L_l5HBnNInwdLEqqdTtfNLPZJ0OBFqjXAptcr1dTQH0OAUGAvAA5O7xv5lRlz2iT7_f3st2UpIz_ufeYyvZ_hnrb2bUqZCUniKLJfUeaTmt5n4Od16aCJBYHSDFtEnOS_pBUSqVwnV5_c19L8N5akCTagx6lGohMUpQDNOZvQymLGoOkRtLOi5ud6j3EdrWHNC0ggrtjzuIaLa2w2I0V2_42jEdDoQxrlePp-rwTkfMzjDnJscV81twVLi0Kyzks7b0M3u_rYI6ExVl8duxn_tT8tTrZ98Lt9CsyPvFglzx6PMJ2oJqYTe-CwWDKkicmx8bF8RyD7i0JI1MtqUXV5b6ss44YhoEu0XT48AG3dK90gDFbN4_uAJDiQmIUkCnvh9KwSTH0sGEbxSCSqGXuXbBFAKUhRziKJ-xBT5ifJlG3Qm1JoQXgpNR6xFHzyEbJtEy0iDeDWJFi6mxtm5m0YH4zEgvpYt-PO6e0yNtdTGMGt30ZRNfvv5rGgISLHpFVxo1HvTqdjcn48veZvqyuNAcam3wWO5I1zW_pXz-QomK1C3CS33VdUdvH-DGPyRMObAM"
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
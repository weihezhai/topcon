import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AGAc9QMVYSGyy6FwtMaD3PFvazIEwh1K9g4CKSsiPuNCP2rtqeKzEydqaEN38-YZiqV6AnxbtOM7T9F8I6zCVkBYE_rmmWgyt2-Isx2ksjPf9oah3xkrQFf117zZoLZ27PRrnx6YOhaTdP45VML8BNTkmpa80FH0JyIiTaL-L54B0NM9k4BmDKCiQW0k4CIEJZK3_tuUffGfbUc1Eoe8ytnfakSUjgCY3jtVjhNcqill3lF7WFb5LQsQyqUNYENVHsxP7B8rnJfhXYq5lnwkCoDV9Bwqsy_Yi_4xRtttSINZhiiGKXvO0QHZaLMRBiGNFO0vyW0_xsVTC5q_qYGvLYXitqZOUB2SEu1bUFr00z3S0JHMZ_BH0K5IoEZYM5HqFYmsgslPNSk9zRubPlzzCuAXj0tnE_-lLIOD2AT036mghhmV1dibJVfv5dTbns3uzyvoG8YY2RyS3mz_Up9QsN51IzbbzM3kv01zjfuGypuUk1HqzoEMdKqF36MJ2bug9tFINUjjLAub-NOvqXBcF8zj7NWEikPEwDGgpCIRFIB2C4zz35GLUXHReTOXasdPH3HNqKclLL2jvv8OMvi_OHa-I_dsFTWr2SjIH9vLHdRMZ_89lDavWpyKdsvbNd9mmlZWT5deKARiroRLiltlH-QgPkR-xTg_RGxe41BCyTN4LjS1gvb5TrGklXavvYCwO47h1CKoSuz2puD148CRfLjpAPw-gMSltnCoT6YGIT84Z_4wix_TV-4RAgx0zqizpdhBCI1I46I8bjlN-D1b7QP6a9f6y6KlmQopzfA6I4_1tm7slYJb6lfOJSDsDEht7Yh-j5Yqlp-m75yKism6Hd_v0Wu64l8Iq1TvuyP9P7uOl0W8nm5yR7iCpc1-Ljj774XmeXkOGReqtX5e4mEVtWiiiOeTmocv7n3aLRqIlv9iOz5giY5DjjtkOyTEz0ek78gboZ0rheIK1_aZQtaU05ysyrwNOg4_BUVe0yWY9phsqxKKLwAdR3gJPgZdjSpE1ZIWBbBSS8tsc7k5jKY9XKjDZJyYy23hWZCywyrTzfH_5pzmavFlVjWWRuaB77YJdFiultlPVmjbLXkJnX93nKVlR-K2dg0GAmLhidIf8nUbF1hzR61i5MTsWe0M8I_FoTvk2cuBKyzHcgXWAygC0blQ8QyX1Hsj7jmzKfJ234E0ytrDlEjUvV4UZiYlK6O6OeL_1AAG69G3K7zTsrAR957P6ddw_j00KcbD0FBVf7hvVlI4K_R3trNiJ1Z1MbD1oSI"
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
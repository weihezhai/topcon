import dropbox
import os
import sys

# --- Configuration ---
# Get the static access token from an environment variable for security
ACCESS_TOKEN = "sl.u.AF_NRG4mqClUESvwyoodN6H-vVQ8jPMvPKlASZegtJkbhqhZaIvenmCO1Nf35qy5T8AJN9zcDnu9eqQ4BKRGKbE9fATJW6F0v06lDh3SgldFQMdsIbpuWPxuFejetpQyqGzUxM7zF4-zUJyRR9k0-NRXsnBl8nsdaMMOmlgTVEOk04PtOWJTNL6RC4OVHvggCZe1wgBZW_SKLMXpPBCihu2EqjL-9iiV-Sliw4bswYPJ3X3ucOgjps2hlOd7ySZzmvB8mGN_6WPST_MYguV5rn_MM_wjrwEtvyKEafTZFlfayBudb0D5LUCaxZvHi8sFcoU9lZbKCUGJdOq8PBwSvZAYRPuXFu3xNYMdeu2J6Ki2AgD2qx-2k-y1mM6hRJsjFoY9IY2O9tP2Xej46khpMDPIfSkBtHMMvJmODvIOVQuhUjdc2z9NOz28ED9ZzWv6F6zOb9bPQPvXHezYK1cq7Q-p5DXlP2EjNxN3YdGjSkZ85ap-6PJkgz4MzhyWWk0SR4Jp7lFaMC9cqoDSAZaIo7vgMii2LmNHRhV4JdpxcdxUm-zKlzOHKXIGZWS3d2b9CHNt0xWEc66wkFREQcFQr_lj_C35aMZpngUx-I3zBs0ZRZizfML5SrpH7SfOkgD5G7CxfobGoQ3eVK1KI0HrkIxWEhIzerOmMCQLbwyIsiChudCQpxYDsvE4LOUrTTZMcnGfuaAqjYHIjtuElF70khd1V7BwgXakliMm-goSEBU7P9xcIFMxLa8BuNUI1UYCuWwoxyhGT1BnlblifAO5cfN3dbGr5ZYs-xQhm5sBSK2LIzMCx8vUJ9jOOk6jBmf9A_hEFqxuIIRF_rR-8uOXH53lyWRTqCVbRw8xfmWzF5BXhpj5JOFN1gvydTkJlAWxF8mSGFvDMWKeupAsCnf9V-PO6VYvMpOciz3Gy8Sy1FiuPxhorxTseEniVe7cJBUckV4At0JEF6cid0rl3DaDlCF3uvdv9fypRRTSPAg8XOAYDiPBeuAj73FS1Om4kH7UnTo6rhXRrI8-eTKb59Wp-hkMkYNkqC29EqpzuZSz03MIld-i-zA1O7fEBVkrrddxd2si2bHOel2F6I4TpZSSs1SIkmKlS599QanTJKidYuYY7K0dZDJ8wHawjazPrO0uqbPb7ufawHUpGAyt9unjkDCAycGA1K_g-PgdNqBKnK-AEg7Esnj_kbRMS5KKYYnGWnMODfgagfQz_Vt2idJWpc9x7Ng6Hws_ss4dQKMhuuyfN26mgmBkhRTwYMJS9BSsJFo"
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
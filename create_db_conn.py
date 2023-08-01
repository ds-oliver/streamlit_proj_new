import os

# Set the project directory path
project_dir = '/Users/hogan/dev/streamlit_proj_new'
streamlit_dir = os.path.join(project_dir, '.streamlit')

# Create the .streamlit directory if it doesn't exist
os.makedirs(streamlit_dir, exist_ok=True)

# Define the path to the db file
db_file_path = '/Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/db_files/players.db'

# Generate the content for secrets.toml
secrets_content = f"""# .streamlit/secrets.toml
[connections.players_db]
url = "sqlite:///{db_file_path}"
"""

# Write the content to secrets.toml
secrets_file_path = os.path.join(streamlit_dir, 'secrets.toml')
with open(secrets_file_path, 'w') as secrets_file:
    secrets_file.write(secrets_content)

print(f"Secrets file created at: {secrets_file_path}")

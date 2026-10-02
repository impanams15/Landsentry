import hashlib

admin_hash = hashlib.sha256('Admin123'.encode('utf-8')).hexdigest()
research_hash = hashlib.sha256('Research123'.encode('utf-8')).hexdigest()
authority_hash = hashlib.sha256('Authority123'.encode('utf-8')).hexdigest()

content = f'''[users.admin]
# Password is: Admin123
password_hash = "{admin_hash}"
role = "Admin"

[users.researcher]
# Password is: Research123
password_hash = "{research_hash}"
role = "Researcher"

[users.authority]
# Password is: Authority123
password_hash = "{authority_hash}"
role = "Authority"
'''

with open('.streamlit/secrets.toml', 'w') as f:
    f.write(content)

print(content)

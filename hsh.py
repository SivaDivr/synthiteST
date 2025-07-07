import streamlit_authenticator as stauth

# Correct way to use Hasher in recent versions
hasher = stauth.Hasher()
hashed_passwords = hasher.hash('P@ssw0rd')

print(hashed_passwords)



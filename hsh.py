import streamlit_authenticator as stauth

# Correct way to use Hasher in recent versions
hasher = stauth.Hasher()
hashed_passwords = hasher.hash('1234')

print(hashed_passwords)



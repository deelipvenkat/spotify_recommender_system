from flask_spotify_auth import getAuth, refreshAuth, getToken

#Add your client ID
# YOU SHOULD USE os.environ['CLIENT']
CLIENT_ID = "d3f0fc7df42e435c8ea8f29bcb89adfc"

#aDD YOUR CLIENT SECRET FROM SPOTIFY
# YOU SHOULD USE os.environ['SECRET']
CLIENT_SECRET = "dac0283d86fd4caba2a848e158b12553"

#Port and callback url can be changed or ledt to localhost:5000
PORT = "5000"
CALLBACK_URL = "http://localhost"
#CALLBACK_URL = "https://automated-credit-system.herokuapp.com/"
#Add needed scope from spotify user
SCOPE = "user-library-read playlist-modify-public"
#token_data will hold authentication header with access code, the allowed scopes, and the refresh countdown
TOKEN_DATA = []


def getUser():
    return getAuth(CLIENT_ID, "{}:{}/callback/".format(CALLBACK_URL, PORT), SCOPE)

def getUserToken(code):
    global TOKEN_DATA
    TOKEN_DATA = getToken(code, CLIENT_ID, CLIENT_SECRET, "{}:{}/callback/".format(CALLBACK_URL, PORT))

def refreshToken(time):
    time.sleep(time)
    TOKEN_DATA = refreshAuth()

def getAccessToken():
    return TOKEN_DATA

import logging
import jwt.algorithms
from rest_framework import authentication
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth import get_user_model
from django.conf import settings
import requests
import jwt
import json

logger = logging.getLogger(__name__)

User = get_user_model()


class Auth0JWTAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.META.get("HTTP_AUTHORIZATION", "")
        if not auth_header.startswith("Bearer "):
            return None
        token = auth_header.split(" ")[1]

        try:
            # Verify the token with Auth0
            header = jwt.get_unverified_header(token)
            jwks = requests.get('{}/.well-known/jwks.json'.format(settings.AUTH0_ISSUER)).json()
            public_key = None
            for jwk in jwks['keys']:
                if jwk['kid'] == header['kid']:
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))

            if public_key is None:
                raise AuthenticationFailed("Public key not found")

            jwt_payload = jwt.decode(token, public_key, audience=settings.AUTH0_CLIENT_ID, issuer=settings.AUTH0_ISSUER, algorithms=['RS256'])

            # Extract user attributes from Auth0 response
            user = User.objects.filter(email__iexact=jwt_payload.get("email")).first()
            
            if not user:
                raise AuthenticationFailed("User not found")
            
            return (user, token)
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise AuthenticationFailed(f"Authentication error: {str(e)}")
    
    def authenticate_header(self, request):
        """
        Return a string to be used as the value of the WWW-Authenticate
        header in a 401 Unauthorized response.
        """
        return "Bearer"








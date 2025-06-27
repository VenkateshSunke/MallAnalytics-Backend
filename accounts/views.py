from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserSerializer, UserUpdateSerializer

# Create your views here.

@api_view(["GET", "PUT"])
@permission_classes([IsAuthenticated])
def user_detail(request):
    user = request.user

    if request.method == "GET":
        serializer = UserSerializer(user)
        return Response(serializer.data)

    elif request.method == "PUT":
        request_body = request.data
        if "email" in request_body:
            # lower case email
            request_body["email"] = request_body["email"].lower()

        serializer = UserUpdateSerializer(user, data=request.data, partial=True, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

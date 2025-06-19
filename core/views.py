from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import *
from .serializers import *
from shapely import wkt
from shapely.geometry import Polygon, Point
import uuid
from datetime import datetime
import boto3
from django.conf import settings
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import default_storage
import re
from rest_framework.pagination import PageNumberPagination
from django.db.models import Q

from decouple import config

# --- API for Screen 1 ---
class UserListView(APIView):
    def get(self, request):
        search = request.query_params.get('search', '')
        page = request.query_params.get('page', 1)
        page_size = request.query_params.get('page_size', 10)

        users = User.objects.all()
        if search:
            users = users.filter(Q(name__icontains=search) | Q(email__icontains=search))

        paginator = PageNumberPagination()
        paginator.page_size = int(page_size)
        result_page = paginator.paginate_queryset(users, request)
        serializer = UserListSerializer(result_page, many=True)
        return paginator.get_paginated_response(serializer.data)

# --- API for Screen 2 ---
class UserDetailView(APIView):
    def get(self, request, user_id):
        try:
            user = User.objects.get(user_id=user_id)
        except User.DoesNotExist:
            return Response({"detail": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        serializer = UserDetailWithVisitsSerializer(user)
        return Response(serializer.data)

# --- API for regestering user ----
class CreateUserView(APIView):
    def post(self, request):
        serializer = UserCreateSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response(UserCreateSerializer(user).data, status=status.HTTP_201_CREATED)
        print("Validation errors:", serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

class PhotoUploadView(APIView):
    parser_classes = [MultiPartParser]  # Important for file uploads

    def post(self, request):
        if 'photo' not in request.FILES:
            return Response({'error': 'No photo provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Extract only the fields needed for UserCreateSerializer
        user_data = {
            'name': request.data.get('name'),
            'email': request.data.get('email'),
            'date_of_birth': request.data.get('date_of_birth'),
            'address': request.data.get('address'),
            'cell_phone': request.data.get('cell_phone'),
            'picture_url': request.data.get('picture_url', ''),  # Optional field
        }

        photo = request.FILES['photo']

        # Validate user data
        serializer = UserCreateSerializer(data=user_data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # If validation passes, upload the photo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Clean the filename to prevent issues
        clean_name = re.sub(r'[^\w\.-]', '_', photo.name)
        filename = f'user_photos/{timestamp}_{clean_name}'
        
        try:
            file_path = default_storage.save(filename, photo)
            file_url = default_storage.url(file_path)
            
            # Update the picture_url in user_data if you want to save it
            user_data['picture_url'] = file_url
            
            # If you want to create the user at this point:
            # user = serializer.save()
            
            return Response({
                'photo_url': file_url,
                'user_data': user_data  # Optional: return the validated data
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class MappingDataView(APIView):
    def get(self, request):
        def parse_polygon(polygon_wkt):
            if polygon_wkt.startswith("SRID="):
                polygon_wkt = polygon_wkt.split(";", 1)[1]
            polygon = wkt.loads(polygon_wkt)
            return list(polygon.exterior.coords)

        stores = [
            {
                "id": store.id,
                "name": store.name,
                "category": store.category,
                "polygon": parse_polygon(store.polygon),
                "is_mapped": store.is_mapped,
            } for store in Store.objects.all()
        ]

        cameras = [
            {
                "id": cam.id,
                "position": cam.position,
                "orientation": cam.orientation,
                "fov_angle": cam.fov_angle,
                "fov_range": cam.fov_range,
            } for cam in Camera.objects.all()
        ]

        calibrations = {
            cal.store_id: cal.matrix for cal in Calibration.objects.all()
        }

        return Response({"stores": stores, "cameras": cameras, "calibration": {"store_matrices": calibrations}})

class ImportMappingView(APIView):
    def post(self, request):
        data = request.data
        for store_id, store in data['stores'].items():
            poly = Polygon(store['polygon'])
            Store.objects.update_or_create(
                id=store_id,
                defaults={
                    "name": store["name"],
                    "category": store["category"],
                    "polygon": f"SRID=4326;{poly.wkt}",
                    "is_mapped": store["is_mapped"],
                }
            )
        for cam_id, cam in data["cameras"].items():
            point = Point(cam["position"])
            Camera.objects.update_or_create(
                id=cam_id,
                defaults={
                    "position": f"SRID=4326;{point.wkt}",
                    "orientation": cam["orientation"],
                    "fov_angle": cam["fov_angle"],
                    "fov_range": cam["fov_range"],
                }
            )
        for store_id, matrix in data["calibration"]["store_matrices"].items():
            Calibration.objects.update_or_create(
                store_id=store_id,
                defaults={"matrix": matrix}
            )
        return Response({"status": "success", "message": "Mapping data imported to DB."})

class CreateVisitView(APIView):
    def post(self, request):
        serializer = VisitSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

class GetVisitView(APIView):
    def get(self, request, visit_id):
        try:
            visit = Visit.objects.get(pk=visit_id)
            return Response(VisitSerializer(visit).data)
        except Visit.DoesNotExist:
            return Response({"detail": "Visit not found"}, status=404)

class CreateMovementView(APIView):
    def post(self, request):
        serializer = UserMovementSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

class MovementByVisitView(APIView):
    def get(self, request, visit_id):
        movements = UserMovement.objects.filter(visit_id=visit_id)
        return Response(UserMovementSerializer(movements, many=True).data)

class CreateListStoreView(APIView):
    def post(self, request):
        serializer = StoreSerializer(data=request.data)
        if serializer.is_valid():
            if MallStore.objects.filter(store_code=serializer.validated_data['store_code']).exists():
                return Response({"detail": "Store already exists"}, status=400)
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

    def get(self, request):
        stores = MallStore.objects.all()
        return Response(StoreSerializer(stores, many=True).data)

class CreateListInterestView(APIView):
    def post(self, request):
        data = request.data.copy()
        data['interest_id'] = str(uuid.uuid4())
        serializer = InterestSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

    def get(self, request):
        interests = Interest.objects.all()
        return Response(InterestSerializer(interests, many=True).data)

class CreateUserInterestView(APIView):
    def post(self, request):
        user_id = request.data['user_id']
        interest_id = request.data['interest_id']
        source = request.data['source']
        if UserInterest.objects.filter(user_id=user_id, interest_id=interest_id).exists():
            return Response({"detail": "User interest already linked"}, status=400)
        UserInterest.objects.create(
            user_id=user_id,
            interest_id=interest_id,
            source=source,
            created_at=datetime.utcnow()
        )
        return Response({"status": "linked"})

class GetUserInterestsView(APIView):
    def get(self, request, user_id):
        interests = UserInterest.objects.filter(user_id=user_id)
        return Response(UserInterestSerializer(interests, many=True).data)

class uploadPhotoView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        photo = request.FILES.get('photo')
        if not photo:
            return Response({"detail": "No photo uploaded"}, status=400)

        s3 = boto3.client('s3',
            aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY')
        )

        bucket_name = config('AWS_STORAGE_BUCKET_NAME')
        filename = f'users/{photo.name}'

        s3.upload_fileobj(photo, bucket_name, filename)
        photo_url = f'https://{bucket_name}.s3.amazonaws.com/{filename}'

        return Response({'photo_url': photo_url}, status=200)
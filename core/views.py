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
from urllib.parse import unquote
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.generics import ListAPIView, CreateAPIView, RetrieveUpdateDestroyAPIView, ListCreateAPIView
from django.db.models import Avg, Count, Q
from decouple import config
from django.utils import timezone
from datetime import timedelta
from django.utils.timezone import now
import pytz
from django.utils.timezone import is_naive
from datetime import timezone

from .utils.sendgrid_service import create_list, sync_contacts_to_list, create_sendgrid_campaign, schedule_sendgrid_campaign, get_senders, get_default_sender_id, SendGridError, get_campaign_details, get_suppression_groups, get_campaign_stats, delete_sendgrid_campaign, delete_sendgrid_list

# --- API for Screen 1 ---
class UserListView(APIView):
    # def get(self, request):
    #     search = request.query_params.get('search', '')
    #     page = request.query_params.get('page', 1)
    #     page_size = request.query_params.get('page_size', 10)

    #     users = User.objects.all().order_by('-created_at')  # or '-date_joined'

    #     if search:
    #         users = users.filter(Q(name__icontains=search) | Q(email__icontains=search))

    #     paginator = PageNumberPagination()
    #     paginator.page_size = int(page_size)
    #     result_page = paginator.paginate_queryset(users, request)
    #     serializer = UserListSerializer(result_page, many=True)
    #     return paginator.get_paginated_response(serializer.data)
    def get(self, request):
        search = request.query_params.get('search', '')
        page = request.query_params.get('page', 1)
        page_size = request.query_params.get('page_size', 10)

        # custom filters
        name = request.query_params.get('name')
        membership = request.query_params.get('membership')
        # store = request.query_params.get('store')
        monthly_freq = request.query_params.get('monthlyFreq')
        last_visit = request.query_params.get('lastVisit')
        visits=request.query_params.get('visits')
        email = request.query_params.get('email')

        users = User.objects.all().order_by('-created_at')

        if search:
            users = users.filter(Q(name__icontains=search) | Q(email__icontains=search))

        if name:
            users = users.filter(name__icontains=name)

        if email:
            users = users.filter(email__icontains=email)

        if membership:
            users = users.filter(pattern_1__icontains=membership)  # or adjust field name

        if monthly_freq:
            try:
                users = users.filter(monthly_freq=int(monthly_freq))
            except ValueError:
                pass

        if last_visit:
            try:
                parsed_date = parse_date(last_visit)
                if parsed_date:
                    users = users.filter(last_visit=parsed_date)
            except ValueError:
                pass
        
        if visits:
            try:
                users = users.filter(life_visits=int(visits))
            except ValueError:
                pass
        # Add more filters (e.g., store) as needed

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
    def delete(self, request, user_id):
        try:
            user = User.objects.get(user_id=user_id)
        except User.DoesNotExist:
            return Response({"detail": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        user.delete()
        return Response({"detail": "User deleted successfully"}, status=status.HTTP_204_NO_CONTENT)

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
@method_decorator(csrf_exempt, name='dispatch')
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
        # filename = f'users/{photo.name}'
        filename = f'users/{uuid.uuid4().hex}_{photo.name}'

        try:
            s3.upload_fileobj(photo, bucket_name, filename)
        except Exception as e:
            return Response({"detail": str(e)}, status=500)

        return Response({"photo_url": filename}, status=200)

        # s3.upload_fileobj(photo, bucket_name, filename)
        # photo_url = f'https://{bucket_name}.s3.amazonaws.com/{filename}'

        # # Register the face in Rekognition
        # rekognition = boto3.client('rekognition',
        #     aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
        #     aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
        #     region_name=config('AWS_REGION')
        # )
        # collection_id = config('AWS_REKOGNITION_COLLECTION_ID')

        # response = rekognition.index_faces(
        #     CollectionId=collection_id,
        #     Image={
        #         'S3Object': {
        #             'Bucket': bucket_name,
        #             'Name': filename
        #         }
        #     },
        #     ExternalImageId=photo.name,  # Optional: used for linking
        #     DetectionAttributes=['DEFAULT']
        # )
        
        # # Extract faceId(s)
        # face_records = response.get('FaceRecords', [])
        # if face_records:
        #     face_id = face_records[0]['Face']['FaceId']
        #     return Response({'photo_url': photo_url, 'face_id': face_id}, status=200)
        # else:
        #     return Response({'photo_url': photo_url, 'detail': 'No face detected'}, status=400)

        # return Response({'photo_url': photo_url}, status=200)

class GeneratePresignedURL(APIView):
    def get(self, request):
        s3_key = request.query_params.get('s3_key')
        if not s3_key:
            return Response({"detail": "Missing s3_key"}, status=400)
        s3_key = unquote(s3_key)
        s3 = boto3.client(
            's3',
            aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
            region_name=config('AWS_REGION')
        )

        try:
            presigned_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': config('AWS_STORAGE_BUCKET_NAME'), 'Key': s3_key},
                ExpiresIn=3600  # 1 hour
            )
            return Response({"url": presigned_url})
        except Exception as e:
            return Response({"detail": str(e)}, status=500)

class BusinessHourListView(ListAPIView):
    queryset = BusinessHour.objects.all()
    serializer_class = BusinessHourSerializer


class EmailCampaignListCreateView(ListAPIView, CreateAPIView):
    queryset = EmailCampaign.objects.all()
    serializer_class = EmailCampaignSerializer

    def perform_create(self, serializer):
        campaign = serializer.save()
        try:
            sendgrid_list_id = create_list(campaign.name)
            campaign.sendgrid_list_id = sendgrid_list_id
            campaign.save()
            print(f"Created SendGrid list {sendgrid_list_id} for campaign {campaign.campaign_id}")
        except Exception as e:
            # Roll back campaign creation if SendGrid list creation fails
            campaign.delete()
            print(f"Failed to create SendGrid list for campaign {campaign.campaign_id}: {e}")
            raise serializers.ValidationError({
                'sendgrid_list_id': f'Failed to create SendGrid list: {e}'
            })


class EmailCampaignDetailView(RetrieveUpdateDestroyAPIView):
    queryset = EmailCampaign.objects.all()
    serializer_class = EmailCampaignSerializer
    lookup_field = 'campaign_id'

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        # Delete all steps and their SendGrid campaigns
        step_results = []
        for step in instance.steps.all():
            sg_id = step.sendgrid_campaign_id
            step.delete()
            if sg_id:
                sg_result = delete_sendgrid_campaign(sg_id)
                step_results.append((sg_id, sg_result))
        # Delete the SendGrid list
        list_result = None
        if instance.sendgrid_list_id:
            list_result = delete_sendgrid_list(instance.sendgrid_list_id)
        # Delete the campaign itself
        super().destroy(request, *args, **kwargs)
        # Build response
        data = {"detail": "Campaign and all steps deleted successfully"}
        if step_results:
            data["steps"] = [
                {"sendgrid_campaign_id": sg_id, "deleted": res} for sg_id, res in step_results
            ]
        if instance.sendgrid_list_id:
            if list_result:
                data["sendgrid_list"] = f"SendGrid list {instance.sendgrid_list_id} deleted successfully."
            else:
                data["sendgrid_list"] = f"Failed to delete SendGrid list {instance.sendgrid_list_id}. It may have already been deleted or there was an error."
        return Response(data, status=204)


class EmailCampaignToggleView(APIView):
    def patch(self, request, campaign_id):
        try:
            campaign = EmailCampaign.objects.get(pk=campaign_id)
            campaign.is_active = request.data.get('is_active', campaign.is_active)
            campaign.save()
            return Response({'status': 'updated', 'is_active': campaign.is_active})
        except EmailCampaign.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)


class AddCampaignContactsView(APIView):
    def post(self, request, campaign_id):
        try:
            campaign = EmailCampaign.objects.get(pk=campaign_id)
            user_ids = request.data.get('user_ids', [])
            
            if not user_ids:
                return Response({'error': 'No user_ids provided'}, status=400)
            
            added = []
            for user_id in user_ids:
                user = User.objects.filter(user_id=user_id).first()
                if user:
                    CampaignContact.objects.get_or_create(campaign=campaign, user=user)
                    added.append(user.user_id)
            
            # Sync contacts to SendGrid only if we have a SendGrid list
            if campaign.sendgrid_list_id:
                try:
                    contacts = CampaignContact.objects.filter(campaign=campaign)
                    status_code, response = sync_contacts_to_list(campaign, contacts)
                    print(f"Synced {len(added)} contacts to SendGrid list {campaign.sendgrid_list_id}")
                except Exception as e:
                    print(f"Failed to sync contacts to SendGrid for campaign {campaign_id}: {e}")
                    # Still return success for the database operation
                    return Response({
                        "added_user_ids": added,
                        "warning": "Contacts added to database but SendGrid sync failed"
                    }, status=status.HTTP_201_CREATED)
            else:
                print(f"Campaign {campaign_id} has no SendGrid list ID, skipping sync")
            
            return Response({"added_user_ids": added}, status=status.HTTP_201_CREATED)
            
        except EmailCampaign.DoesNotExist:
            return Response({'error': 'Campaign not found'}, status=404)


class CampaignMinimalListView(ListAPIView):
    queryset = EmailCampaign.objects.all()
    serializer_class = CampaignMinimalSerializer


class CampaignContactListView(ListAPIView):
    serializer_class = CampaignContactSerializer

    def get_queryset(self):
        campaign_id = self.kwargs['campaign_id']
        return CampaignContact.objects.filter(campaign__campaign_id=campaign_id)

class DashboardMetricsView(APIView):
    def get(self, request):
        now = timezone.now()
        seven_days_ago = now - timedelta(days=7)
        total_visitors = User.objects.count()
        active_users = User.objects.filter(
            visits__start_time__isnull=False,
            visits__end_time__isnull=True
        ).distinct().count()

        new_active_users = User.objects.filter(first_visit__gte=seven_days_ago).count()
        avg_duration = Visit.objects.exclude(duration__isnull=True).aggregate(avg=Avg("duration"))["avg"]

        if avg_duration:
            # Format timedelta to HH:MM:SS
            hours, remainder = divmod(avg_duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            avg_visit_duration = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        else:
            avg_visit_duration = None

        return Response({
            "total_visitors": total_visitors,
            "active_users": active_users,
            "avg_visit_duration": avg_visit_duration,
            "new_active_users": new_active_users
        })

class CampaignStepListCreateView(ListCreateAPIView):
    serializer_class = CampaignStepSerializer
    
    def get_queryset(self):
        campaign_id = self.kwargs['campaign_id']
        return CampaignStep.objects.filter(campaign__campaign_id=campaign_id)
    
    def create(self, request, *args, **kwargs):
        try:
            return super().create(request, *args, **kwargs)
        except Exception as e:
            print(f"Error creating campaign step: {e}")
            return Response({'error': str(e)}, status=400)
    
    def perform_create(self, serializer):
        campaign_id = self.kwargs['campaign_id']
        try:
            campaign = EmailCampaign.objects.get(campaign_id=campaign_id)
        except EmailCampaign.DoesNotExist:
            raise serializers.ValidationError(f"Campaign with ID {campaign_id} does not exist")
        
        step = serializer.save(campaign=campaign)
        
        # Validate step content before creating SendGrid campaign
        if not step.subject or not step.subject.strip():
            print(f"Step {step.id} created without subject")
            return
        
        if not step.body or not step.body.strip():
            print(f"Step {step.id} created without body")
            return
        
        # Get sender_id and suppression_group_id from request data
        request = self.request
        sender_id = request.data.get('sender_id') or get_default_sender_id()
        suppression_group_id = request.data.get('suppression_group_id')
        
        if sender_id and campaign.sendgrid_list_id:
            try:
                # Validate sender_id
                if not str(sender_id).isdigit():
                    print(f"Invalid sender_id: {sender_id}")
                    return
                
                sendgrid_campaign_id = create_sendgrid_campaign(step, sender_id, suppression_group_id)
                if sendgrid_campaign_id:
                    step.sendgrid_campaign_id = sendgrid_campaign_id
                    step.save()
                    print(f"Created SendGrid campaign {sendgrid_campaign_id} for step {step.id}")
                    # Schedule the campaign immediately if send_at is set
                    if step.send_at:
                        try:
                            send_at = step.send_at
                            if is_naive(send_at):
                                local_tz = pytz.timezone('Asia/Kolkata')
                                send_at = local_tz.localize(send_at)
                            send_at_utc = send_at.astimezone(timezone.utc)
                            print(f"Scheduling SendGrid campaign {sendgrid_campaign_id}: original send_at={step.send_at}, UTC={send_at_utc}")
                            schedule_sendgrid_campaign(sendgrid_campaign_id, send_at_utc)
                            print(f"Scheduled SendGrid campaign {sendgrid_campaign_id} for {send_at_utc}")
                        except Exception as e:
                            print(f"Failed to schedule SendGrid campaign {sendgrid_campaign_id}: {e}")
            except Exception as e:
                print(f"Failed to create SendGrid campaign for step {step.id}: {e}")
                # Don't raise exception here - step is still created, just without SendGrid campaign

class CampaignStepDetailView(RetrieveUpdateDestroyAPIView):
    queryset = CampaignStep.objects.all()
    serializer_class = CampaignStepSerializer
    lookup_field = 'pk'

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        sendgrid_id = instance.sendgrid_campaign_id
        response = super().destroy(request, *args, **kwargs)
        sg_result = None
        if sendgrid_id:
            sg_result = delete_sendgrid_campaign(sendgrid_id)
        # Build a custom response
        data = {"detail": "Step deleted successfully"}
        if sendgrid_id:
            if sg_result:
                data["sendgrid"] = f"SendGrid campaign {sendgrid_id} deleted successfully."
            else:
                data["sendgrid"] = f"Step deleted, but failed to delete SendGrid campaign {sendgrid_id}. It may have already been deleted or there was an error."
        return Response(data, status=204)

# ScheduleCampaignStepView now allows selecting the sender by passing 'sender_id' in the POST body.
class ScheduleCampaignStepView(APIView):
    def post(self, request, step_id):
        try:
            step = CampaignStep.objects.get(pk=step_id)
            campaign = step.campaign
            
            # Validate required data
            if not step.send_at:
                return Response({'error': 'Step must have a send_at time'}, status=400)
            
            # Validate step content
            if not step.subject or not step.subject.strip():
                return Response({'error': 'Step must have a subject'}, status=400)
            
            if not step.body or not step.body.strip():
                return Response({'error': 'Step must have body content'}, status=400)
            
            # Check if send_at is in the future
            if step.send_at <= now():
                return Response({'error': 'Send time must be in the future'}, status=400)
            
            # Get sender_id from request or use default
            sender_id = request.data.get('sender_id')
            if not sender_id:
                sender_id = get_default_sender_id()
                if not sender_id:
                    return Response({'error': 'No sender ID available'}, status=400)
            
            # Validate sender_id
            try:
                sender_id = int(sender_id)
            except (ValueError, TypeError):
                return Response({'error': 'Invalid sender ID format'}, status=400)
            
            suppression_group_id = request.data.get('suppression_group_id')
            
            # Ensure campaign has SendGrid list
            if not campaign.sendgrid_list_id:
                return Response({'error': 'Campaign must have a SendGrid list before scheduling'}, status=400)
            
            # Create SendGrid campaign if not already created
            if not step.sendgrid_campaign_id:
                try:
                    sg_campaign_id = create_sendgrid_campaign(step, sender_id, suppression_group_id)
                    step.sendgrid_campaign_id = sg_campaign_id
                    step.save()
                    print(f"Created SendGrid campaign {sg_campaign_id} for step {step_id}")
                except SendGridError as e:
                    print(f"SendGrid API error creating campaign for step {step_id}: {e}")
                    return Response({'error': f'SendGrid API error: {str(e)}'}, status=500)
                except Exception as e:
                    print(f"Unexpected error creating SendGrid campaign for step {step_id}: {e}")
                    return Response({'error': 'Failed to create SendGrid campaign'}, status=500)
            
            # Schedule the campaign
            try:
                status_code = schedule_sendgrid_campaign(step.sendgrid_campaign_id, step.send_at)
                print(f"Scheduled SendGrid campaign {step.sendgrid_campaign_id} for step {step_id}")
                
                # Get campaign details for debugging
                campaign_details = get_campaign_details(step.sendgrid_campaign_id)
                print(f"Campaign details: {campaign_details}")
                
                return Response({
                    'status': 'scheduled',
                    'sendgrid_campaign_id': step.sendgrid_campaign_id,
                    'send_at': step.send_at.isoformat(),
                    'campaign_details': campaign_details
                })
            except SendGridError as e:
                print(f"SendGrid API error scheduling campaign for step {step_id}: {e}")
                return Response({'error': f'SendGrid API error: {str(e)}'}, status=500)
            except Exception as e:
                print(f"Unexpected error scheduling SendGrid campaign for step {step_id}: {e}")
                return Response({'error': 'Failed to schedule campaign'}, status=500)
                
        except CampaignStep.DoesNotExist:
            return Response({'error': 'Step not found'}, status=404)


class SendGridSenderListView(APIView):
    def get(self, request):
        try:
            senders = get_senders()
            return Response(senders)
        except Exception as e:
            print(f"Failed to get SendGrid senders: {e}")
            return Response({'error': 'Failed to retrieve senders'}, status=500)

class SendGridCampaignDetailsView(APIView):
    """Debug view to get campaign details from SendGrid"""
    def get(self, request, campaign_id):
        try:
            details = get_campaign_details(campaign_id)
            if details:
                return Response(details)
            else:
                return Response({'error': 'Campaign not found'}, status=404)
        except Exception as e:
            print(f"Failed to get SendGrid campaign details: {e}")
            return Response({'error': 'Failed to retrieve campaign details'}, status=500)

class SendGridSuppressionGroupListView(APIView):
    def get(self, request):
        try:
            groups = get_suppression_groups()
            return Response(groups)
        except Exception as e:
            print(f"Failed to get SendGrid suppression groups: {e}")
            return Response({'error': 'Failed to retrieve suppression groups'}, status=500)

class CampaignStepSendGridStatsView(APIView):
    def get(self, request, step_id):
        try:
            step = CampaignStep.objects.get(pk=step_id)
            if not step.sendgrid_campaign_id:
                return Response({'error': 'No SendGrid campaign ID for this step'}, status=404)
            stats = get_campaign_stats(step.sendgrid_campaign_id)
            if stats is None:
                return Response({'error': 'Failed to fetch stats from SendGrid'}, status=500)
            return Response(stats)
        except CampaignStep.DoesNotExist:
            return Response({'error': 'Step not found'}, status=404)
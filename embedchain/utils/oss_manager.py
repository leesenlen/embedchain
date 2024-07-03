import os
import io
from enum import Enum
import oss2
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_random_exponential
from urllib.parse import urlparse

OSS = {
    'endpoint': os.getenv('OSS_ENDPOINT',''),
    'access_key_id': os.getenv('OSS_ACCESS_KEY_ID',''),
    'access_key_secret': os.getenv('OSS_ACCESS_KEY_SECRET',''),
    'bucket_name': os.getenv('OSS_BUCKET_NAME','aiagent-files'),
}


class OSSFileDirectory(Enum):
    CHAT_ATTACHMENTS = 'chat_attachments'
    KNOWLEDGES = 'knowledges'
    GENERATION = 'generation'
    MODELS = 'models'
    LAYOUT_IMG = 'layout_img'

    def __str__(self) -> str:
        return self.value


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class OSSClient:
    def __init__(self,
                 endpoint: str = OSS['endpoint'],
                 access_key_id: str = OSS['access_key_id'],
                 access_key_secret: str = OSS['access_key_secret'],
                 bucket_name: str = OSS['bucket_name']):
        self.endpoint = endpoint
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.bucket_name = bucket_name
        # 初始化认证
        auth = oss2.Auth(access_key_id, access_key_secret)
        # 初始化Bucket
        self.client = oss2.Bucket(auth, endpoint, bucket_name)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(1))
    def upload_file(self, local_file_path, prefix=OSSFileDirectory.CHAT_ATTACHMENTS):
        if not os.path.exists(local_file_path):
            raise Exception(f"{local_file_path}文件不存在")
        if not prefix in [member for member in OSSFileDirectory]:
            raise Exception(f"{prefix}目录不存在")
        try:
            remote_file_path = self.generate_remote_file_path(local_file_path, prefix)
            self.client.put_object_from_file(remote_file_path, local_file_path)
            return self.build_image_url(remote_file_path)
        except Exception as e:
            raise Exception(f"上传文件失败: {e}")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(1))
    def upload_pil_image_to_oss(self, image, prefix, name):
        """
        Upload a PIL.Image.Image object to OSS.

        Args:
            image (PIL.Image.Image): The image to upload.
            bucket_name (str): The name of the OSS bucket.
            object_name (str): The name of the object in the OSS bucket.
            access_key_id (str): The access key ID for OSS.
            access_key_secret (str): The access key secret for OSS.
            endpoint (str): The OSS endpoint.

        Returns:
            str: The URL of the uploaded image.
        """
        # 将 PIL 图像转换为字节流
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='jpeg')  # 你可以根据需要更改图像格式
        img_byte_arr = img_byte_arr.getvalue()
        file_name = name + '_' + str(int(datetime.now().timestamp() * 1000)) + ".jpeg"
        try:
            remote_file_path = self.generate_remote_file_path(file_name, prefix)
            self.client.put_object(remote_file_path, img_byte_arr)
            return self.build_image_url(remote_file_path)
        except Exception as e:
            raise Exception(f"上传文件失败: {e}")

    def generate_remote_file_path(self, local_file_path, prefix=OSSFileDirectory.CHAT_ATTACHMENTS):
        # 获取当前时间
        current_time = datetime.now().strftime("%Y/%m/%d/%H")
        # 获取文件名
        file_name = os.path.basename(local_file_path)
        # 构建远程文件路径
        remote_file_path = f"{prefix}/{current_time}/{file_name}"
        return remote_file_path

    def build_image_url(self, remote_file_path):
        domain = self.bucket_name + '.' + urlparse(self.endpoint).netloc
        return "https://" + domain + "/" + remote_file_path

    def delete_file(self, remote_file_path):
        self.client.delete_object(remote_file_path)




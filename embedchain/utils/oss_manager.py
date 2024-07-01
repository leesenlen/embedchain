import oss2
from aiagent.config.system import OSS
import os
from datetime import datetime
from aiagent.common.enum import OSSFileDirectory
from tenacity import retry, stop_after_attempt, wait_random_exponential
from aiagent.utils.image import download_image,extract_base64_image_data
from aiagent.utils.common import extract_filename
from urllib.parse import urlparse


class OSSManagement:
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
    def upload_file(self, local_file_path, prefix=OSSFileDirectory.CHAT_ATTACHMENTS.value):
        if not os.path.exists(local_file_path):
            raise Exception(f"{local_file_path}文件不存在")
        if not prefix in [member.value for member in OSSFileDirectory]:
            raise Exception(f"{prefix}目录不存在")
        try:
            remote_file_path = self.generate_remote_file_path(local_file_path,prefix)
            self.client.put_object_from_file(remote_file_path, local_file_path)
            return self.build_image_url(remote_file_path)
        except Exception as e:
            raise Exception(f"上传文件失败: {e}")

    def generate_remote_file_path(self, local_file_path, prefix=OSSFileDirectory.CHAT_ATTACHMENTS.value):
        # 获取当前时间
        current_time = datetime.now().strftime("%Y/%m/%d/%H")
        # 获取文件名
        file_name = os.path.basename(local_file_path)
        # 构建远程文件路径
        remote_file_path = f"{prefix}/{current_time}/{file_name}"
        return remote_file_path


    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(1))
    def upload_file_from_base64(self, base64, prefix=OSSFileDirectory.CHAT_ATTACHMENTS.value):
        if not prefix in [member.value for member in OSSFileDirectory]:
            raise Exception(f"{prefix}目录不存在")
        image_extension, image_data = extract_base64_image_data(base64)
        file_name = str(int(datetime.now().timestamp() * 1000))+'.'+image_extension
        try:
            remote_file_path = self.generate_remote_file_path(file_name,prefix)
            self.client.put_object(remote_file_path, image_data)
            return self.build_image_url(remote_file_path)
        except Exception as e:
            raise Exception(f"上传文件失败: {e}")


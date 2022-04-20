# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import sys
import time
import shutil
import hashlib
from io import StringIO, BytesIO
from functools import lru_cache
from contextlib import contextmanager
from typing import List, Union
from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from .logger import logger


class IO:
    @staticmethod
    def register(options):
        pass

    def open(self, path: str, mode: str = 'r', encoding: str = 'utf-8'):
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def move(self, src: str, dst: str):
        raise NotImplementedError

    def copy(self, src: str, dst: str):
        raise NotImplementedError

    def copytree(self, src: str, dst: str):
        raise NotImplementedError

    def makedirs(self, path: str, exist_ok=True):
        raise NotImplementedError

    def remove(self, path: str):
        raise NotImplementedError

    def rmtree(self, path: str):
        raise NotImplementedError

    def listdir(self, path: str, recursive=False, full_path=False, contains=None):
        raise NotImplementedError

    def isdir(self, path: str) -> bool:
        raise NotImplementedError

    def isfile(self, path: str) -> bool:
        raise NotImplementedError

    def abspath(self, path: str) -> str:
        raise NotImplementedError

    def last_modified(self, path: str) -> datetime:
        raise NotImplementedError

    def last_modified_str(self, path: str) -> str:
        raise NotImplementedError

    def size(self, path: str) -> int:
        raise NotImplementedError

    def md5(self, path: str) -> str:
        hash_md5 = hashlib.md5()
        with self.open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    re_remote = re.compile(r'(oss|https?)://')

    def islocal(self, path: str) -> bool:
        return not self.re_remote.match(path.lstrip())

    def is_writable(self, path):
        new_dir = ''
        if self.islocal(path) and not self.exists(path):
            new_dir = path
            while True:
                parent = os.path.dirname(new_dir)
                if self.exists(parent):
                    break
                new_dir = parent
            self.makedirs(path)
        flag = self._is_writable(path)
        if new_dir and self.exists(new_dir):
            self.remove(new_dir)
        return flag

    @lru_cache(maxsize=8)
    def _is_writable(self, path):
        import oss2
        try:
            tmp_file = os.path.join(path, f'.tmp.{time.time()}')
            with self.open(tmp_file, 'w') as f:
                f.write('test line.')
            self.remove(tmp_file)
        except (OSError, oss2.exceptions.RequestError, oss2.exceptions.ServerError):
            return False
        return True


class DefaultIO(IO):
    __name__ = 'DefaultIO'

    def _check_path(self, path):
        if not self.islocal(path):
            raise RuntimeError(
                'OSS Credentials must be provided to use oss_io_config. ')

    def open(self, path, mode='r', encoding="utf-8"):
        self._check_path(path)
        path = self.abspath(path)
        if mode.endswith('b'):
            return open(path, mode=mode)
        else:
            return open(path, mode=mode, encoding=encoding)

    def exists(self, path):
        self._check_path(path)
        path = self.abspath(path)
        return os.path.exists(path)

    def move(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        if src == dst:
            return
        shutil.move(src, dst)

    def copy(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        try:
            shutil.copyfile(src, dst)
        except shutil.SameFileError:
            pass

    def copytree(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src).rstrip('/')
        dst = self.abspath(dst).rstrip('/')
        if src == dst:
            return
        self.makedirs(dst)
        created_dir = {dst}
        for file in self.listdir(src, recursive=True):
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            dst_dir = os.path.dirname(dst_file)
            if dst_dir not in created_dir:
                self.makedirs(dst_dir)
                created_dir.add(dst_dir)
            self.copy(src_file, dst_file)

    def makedirs(self, path, exist_ok=True):
        self._check_path(path)
        path = self.abspath(path)
        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path):
        self._check_path(path)
        path = self.abspath(path)
        if os.path.isdir(path):
            self.rmtree(path)
        else:
            os.remove(path)

    def rmtree(self, path):
        shutil.rmtree(path)

    def listdir(self, path, recursive=False, full_path=False, contains: Union[str, List[str]] = None):
        self._check_path(path)
        path = self.abspath(path)
        if isinstance(contains, str):
            contains = [contains]
        elif not contains:
            contains = ['']
        if recursive:
            files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
            if not full_path:
                prefix_len = len(path.rstrip('/')) + 1
                files = [file[prefix_len:] for file in files]
        else:
            files = os.listdir(path)
            if full_path:
                files = [os.path.join(path, file) for file in files]
        files = [file for file in files if any(keyword in file for keyword in contains)]
        return files

    def isdir(self, path):
        self._check_path(path)
        return os.path.isdir(path)

    def isfile(self, path):
        self._check_path(path)
        return os.path.isfile(path)

    def abspath(self, path):
        self._check_path(path)
        return os.path.abspath(path)

    def last_modified(self, path):
        return datetime.fromtimestamp(float(self.last_modified_str(path)))

    def last_modified_str(self, path):
        self._check_path(path)
        return str(os.path.getmtime(path))

    def size(self, path: str) -> int:
        return os.stat(path).st_size


class OSSIO(DefaultIO):
    "Mixed IO module to support both system-level and OSS IO methods"
    __name__ = 'OSSIO'

    def __init__(self,
                 access_key_id: str,
                 access_key_secret: str,
                 hosts: Union[str, List[str]],
                 buckets: Union[str, List[str]]):

        from oss2 import Auth, Bucket, ObjectIterator
        super().__init__()
        self.ObjectIterator = ObjectIterator
        self.auth = Auth(access_key_id, access_key_secret)
        if isinstance(buckets, str):
            buckets = [buckets]
        if isinstance(hosts, str):
            hosts = [hosts for i in range(len(buckets))]
        else:
            assert len(hosts) == len(buckets), 'number of hosts and number of buckets should be the same'
        self.buckets = {
            bucket_name: Bucket(self.auth, host, bucket_name)
            for host, bucket_name in zip(hosts, buckets)
        }
        self.oss_pattern = re.compile(r'oss://([^/]+)/(.+)')

    def _split_name(self, path):
        m = self.oss_pattern.match(path)
        if not m:
            raise IOError(f'invalid oss path: "{path}", should be "oss://<bucket_name>/path"')
        bucket_name, path = m.groups()
        path = path.replace('//', '/')
        return bucket_name, path

    def _split(self, path):
        bucket_name, path = self._split_name(path)
        try:
            bucket = self.buckets[bucket_name]
        except KeyError:
            raise IOError(f'Bucket {bucket_name} not registered in oss_io_config')
        return bucket, path

    def open(self, full_path, mode='r', encoding='utf-8'):
        if not full_path.startswith('oss://'):
            return super().open(full_path, mode)

        bucket, path = self._split(full_path)
        with mute_stderr():
            path_exists = bucket.object_exists(path)
        if 'w' in mode:
            if path_exists:
                bucket.delete_object(path)
            if 'b' in mode:
                return BinaryOSSFile(bucket, path)
            return OSSFile(bucket, path)
        elif mode == 'a':
            position = bucket.head_object(path).content_length if path_exists else 0
            return OSSFile(bucket, path, position=position)
        else:
            if not path_exists:
                raise FileNotFoundError(full_path)
            obj = bucket.get_object(path)
            # # auto cache large files to avoid memory issues
            # if obj.content_length > 200 * 1024 ** 2:  # 200M
            #     path = cache_file(full_path)
            #     return super().open(path, mode)

            if obj.content_length > 200 * 1024 ** 2:  # 200M
                with tqdm(total=obj.content_length, unit='B', unit_scale=True, unit_divisor=1024, leave=False,
                          desc='reading ' + os.path.basename(full_path)) as t:
                    obj = CallbackIOWrapper(t.update, obj, "read")
                    data = obj.read()
            else:
                import time
                data = obj.read()
            if mode == 'rb':
                return NullContextWrapper(BytesIO(data))
            else:
                assert mode == 'r'
                return NullContextWrapper(StringIO(data.decode()))

    def exists(self, path):
        if not path.startswith('oss://'):
            return super().exists(path)

        bucket, _path = self._split(path)
        if not path.endswith('/'):
            # if file exists
            exists = self._obj_exists(bucket, _path)
        else:
            try:
                self.listdir(path)
                exists = True
            except FileNotFoundError:
                exists = False
        return exists

    def _obj_exists(self, bucket, path):
        with mute_stderr():
            return bucket.object_exists(path)

    def move(self, src, dst):
        if not src.startswith('oss://') and not dst.startswith('oss://'):
            return super().move(src, dst)
        if src == dst:
            return
        self.copy(src, dst)
        self.remove(src)

    def copy(self, src, dst):
        raw_src, raw_dst = str(src), str(dst)
        try:
            cloud_src = src.startswith('oss://')
            cloud_dst = dst.startswith('oss://')
            if not cloud_src and not cloud_dst:
                return super().copy(src, dst)

            if src == dst:
                return
            # download
            if cloud_src and not cloud_dst:
                target_dir, _ = os.path.split(dst)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                bucket, src = self._split(src)
                obj = bucket.get_object(src)
                if obj.content_length > 100 * 1024 ** 2:  # 100M
                    with oss_progress('downloading') as callback:
                        bucket.get_object_to_file(src, dst, progress_callback=callback)
                else:
                    bucket.get_object_to_file(src, dst)
                return
            bucket, dst = self._split(dst)
            # upload
            if cloud_dst and not cloud_src:
                src_size = os.stat(src).st_size
                if src_size > 5 * 1024 ** 3:  # 5G
                    raise RuntimeError(f'A file > 5G cannot be uploaded to OSS. Please split your file first.\n{src}')
                if src_size > 100 * 1024 ** 2:  # 100M
                    with oss_progress('uploading') as callback:
                        bucket.put_object_from_file(dst, src, progress_callback=callback)
                else:
                    bucket.put_object_from_file(dst, src)
                return
            # copy between oss paths
            src_bucket, src = self._split(src)
            total_size = src_bucket.head_object(src).content_length
            if src_bucket.get_bucket_location().location != bucket.get_bucket_location().location:
                import tempfile
                local_tmp = os.path.join(tempfile.gettempdir(), src)
                self.copy(f'oss://{src_bucket.bucket_name}/{src}', local_tmp)
                self.copy(local_tmp, f'oss://{bucket.bucket_name}/{dst}')
                self.remove(local_tmp)
                return

            if total_size < 1024 ** 3 or src_bucket != bucket:  # 1GB
                bucket.copy_object(src_bucket.bucket_name, src, dst)
            else:
                # multipart copy
                from oss2.models import PartInfo
                from oss2 import determine_part_size
                part_size = determine_part_size(total_size, preferred_size=100 * 1024)
                upload_id = bucket.init_multipart_upload(dst).upload_id
                parts = []

                part_number = 1
                offset = 0
                while offset < total_size:
                    num_to_upload = min(part_size, total_size - offset)
                    byte_range = (offset, offset + num_to_upload - 1)

                    result = bucket.upload_part_copy(bucket.bucket_name, src, byte_range, dst, upload_id,
                                                     part_number)
                    parts.append(PartInfo(part_number, result.etag))

                    offset += num_to_upload
                    part_number += 1

                bucket.complete_multipart_upload(dst, upload_id, parts)
        except Exception as e:
            print("haha")
            print("{}".format(e))
            print("Copy failed because oss auth not fully opened. Using first download then upload...")
            try:
                self.download(raw_src, ".easy_distill_tmp_file")
                self.upload(".easy_distill_tmp_file", raw_dst)
                print("Copying done")
            except Exception as e:
                print("{}".format(e))

    def copytree(self, src, dst):
        cloud_src = src.startswith('oss://')
        cloud_dst = dst.startswith('oss://')
        if not cloud_src and not cloud_dst:
            return super().copytree(src, dst)
        if cloud_dst:
            src_files = self.listdir(src, recursive=True)
            max_len = min(max(map(len, src_files)), 50)
            with tqdm(src_files, desc='uploading', leave=False) as progress:
                for file in progress:
                    progress.set_postfix({'file': f'{file:-<{max_len}}'[:max_len]})
                    self.copy(os.path.join(src, file), os.path.join(dst, file))
        else:
            assert cloud_src and not cloud_dst
            self.makedirs(dst)
            created_dir = {dst}
            src_files = self.listdir(src, recursive=True)
            max_len = min(max(map(len, src_files)), 50)
            with tqdm(src_files, desc='downloading', leave=False) as progress:
                for file in progress:
                    src_file = os.path.join(src, file)
                    dst_file = os.path.join(dst, file)
                    dst_dir = os.path.dirname(dst_file)
                    if dst_dir not in created_dir:
                        self.makedirs(dst_dir)
                        created_dir.add(dst_dir)
                    progress.set_postfix({'file': f'{file:-<{max_len}}'[:max_len]})
                    self.copy(src_file, dst_file)

    def listdir(self, path, recursive=False, full_path=False, contains: Union[str, List[str]] = None):
        if not path.startswith('oss://'):
            return super().listdir(path, recursive, full_path, contains)
        if isinstance(contains, str):
            contains = [contains]
        elif not contains:
            contains = ['']

        bucket, path = self._split(path)
        path = path.rstrip('/') + '/'
        files = [obj.key for obj in self.ObjectIterator(bucket, prefix=path, delimiter='' if recursive else '/')]
        try:
            files.remove(path)
        except ValueError:
            pass
        if not files:
            if not self.isdir(path):
                raise FileNotFoundError(f'No such directory: oss://{bucket.bucket_name}/{path}')
        if full_path:
            files = [f'oss://{bucket.bucket_name}/{file}' for file in files]
        else:
            files = [file[len(path):] for file in files]
        files = [file for file in files if any(keyword in file for keyword in contains)]
        return files

    def _remove_obj(self, path):
        bucket, path = self._split(path)
        with mute_stderr():
            bucket.delete_object(path)

    def remove(self, path):
        if not path.startswith('oss://'):
            return super().remove(path)

        if self.isfile(path):
            self._remove_obj(path)
        else:
            return self.rmtree(path)

    def rmtree(self, path):
        if not path.startswith('oss://'):
            return super().rmtree(path)
        # have to delete its content first before delete the directory itself
        for file in self.listdir(path, recursive=True, full_path=True):
            print(f'delete {file}')
            self._remove_obj(file)
        if self.exists(path):
            # remove the directory itself
            if not path.endswith('/'):
                path += '/'
            self._remove_obj(path)

    def makedirs(self, path, exist_ok=True):
        # there is no need to create directory in oss
        if not path.startswith('oss://'):
            return super().makedirs(path)

    def isdir(self, path):
        if not path.startswith('oss://'):
            return super().isdir(path)
        return self.exists(path.rstrip('/') + '/')

    def isfile(self, path):
        if not path.startswith('oss://'):
            return super().isdir(path)
        return self.exists(path) and not self.isdir(path)

    def abspath(self, path):
        if not path.startswith('oss://'):
            return super().abspath(path)
        return path

    def authorize(self, path):
        if not path.startswith('oss://'):
            raise ValueError('Only oss path can use "authorize"')
        import oss2
        bucket, path = self._split(path)
        bucket.put_object_acl(path, oss2.OBJECT_ACL_PUBLIC_READ)

    def last_modified(self, path):
        if not path.startswith('oss://'):
            return super().last_modified(path)
        return datetime.strptime(self.last_modified_str(path), r'%a, %d %b %Y %H:%M:%S %Z') + timedelta(hours=8)

    def last_modified_str(self, path):
        if not path.startswith('oss://'):
            return super().last_modified_str(path)
        bucket, path = self._split(path)
        return bucket.get_object_meta(path).headers['Last-Modified']

    def size(self, path: str) -> int:
        if not path.startswith('oss://'):
            return super().size(path)
        bucket, path = self._split(path)
        return int(bucket.get_object_meta(path).headers['Content-Length'])

    def download(self, oss_path, local_path):
        bucket, path = self._split(oss_path)
        bucket.get_object_to_file(path, local_path)

    def upload(self, local_path, oss_path):
        bucket, path = self._split(oss_path)
        bucket.put_object_from_file(path, local_path)

@contextmanager
def oss_progress(desc):
    progress = None

    def callback(i, n):
        nonlocal progress
        if progress is None:
            progress = tqdm(total=n, unit='B', unit_scale=True, unit_divisor=1024,
                            leave=False, desc=desc, mininterval=1.0, maxinterval=5.0)
        progress.update(i - progress.n)

    yield callback
    if progress is not None:
        progress.close()


def parse_oss_buckets(buckets):
    if "http" in buckets:
        import requests
        import traceback
        try:
            r = requests.get(buckets, allow_redirects=True)
            bucket_name = None
            endpoint = None
            access_key_id = None
            access_key_secret = None
            for line in r.content.decode("utf-8").split("\n"):
                if not line.strip():
                    continue
                key, val = line.strip().split("=")
                if key == "bucket":
                    bucket_name = val
                elif key == "host":
                    endpoint = val
                elif key == "access_key_id":
                    access_key_id = val
                elif key == "access_key_secret":
                    access_key_secret = val
                else:
                    raise RuntimeError
            assert bucket_name is not None and endpoint is not None and \
                   access_key_id is not None and access_key_secret is not None
        except Exception:
            traceback.print_exc()
            raise RuntimeError("Fetch AK file %s failed" % buckets)
    else:
        bucket_name, role_arn_and_host = buckets.split("?")
        access_key_id, access_key_secret, endpoint = role_arn_and_host.split("&")
        bucket_name = bucket_name.replace("oss://", "").strip("'").strip('"').split("/")[0]
        access_key_id = access_key_id.split("=")[-1]
        access_key_secret = access_key_secret.split("=")[-1]
        endpoint = "http://" + endpoint.split("=")[-1]
    return access_key_id, access_key_secret, [endpoint], [bucket_name]


class OSSFile:
    def __init__(self, bucket, path, position=0):
        self.position = position
        self.bucket = bucket
        self.path = path
        self.buffer = StringIO()

    def write(self, content):
        # without a "with" statement, the content is written immediately without buffer
        # when writing a large batch of contents at a time, this will be quite slow
        import oss2
        buffer = self.buffer.getvalue()
        if buffer:
            content = buffer + content
            self.buffer.close()
            self.buffer = StringIO()
        try:
            result = self.bucket.append_object(self.path, self.position, content)
            self.position = result.next_position
        except oss2.exceptions.PositionNotEqualToLength:
            raise RuntimeError(
                f'Race condition detected. It usually means multiple programs were writing to the same file'
                f'oss://{self.bucket.bucket_name}/{self.path} (Error 409: PositionNotEqualToLength)')
        except (oss2.exceptions.RequestError, oss2.exceptions.ServerError) as e:
            self.buffer.write(content)
            logger.info(str(e) + f'when writing to oss://{self.bucket.bucket_name}/{self.path}. Content buffered.')
            raise RuntimeError

    def flush(self, retry=0):
        import oss2
        try:
            self.bucket.append_object(self.path, self.position, self.buffer.getvalue())
        except oss2.exceptions.RequestError as e:
            if 'timeout' not in str(e) or retry > 2:
                raise
            # retry if timeout
            logger.info('| OSSIO timeout. Retry uploading...')
            import time
            time.sleep(5)
            self.flush(retry + 1)
        except oss2.exceptions.ObjectNotAppendable as e:
            from . import io
            logger.info(str(e) + '\nTrying to recover..\n')
            full_path = f'oss://{self.bucket.bucket_name}/{self.path}'
            with io.open(full_path) as f:
                prev_content = f.read()
            io.remove(full_path)
            self.position = 0
            content = self.buffer.getvalue()
            self.buffer.close()
            self.buffer = StringIO()
            self.write(prev_content)
            self.write(content)

    def close(self):
        self.flush()

    def seek(self, position):
        self.position = position

    def __enter__(self):
        return self.buffer

    def __exit__(self, *args):
        self.flush()


class BinaryOSSFile:
    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path
        self.buffer = BytesIO()

    def __enter__(self):
        return self.buffer

    def __exit__(self, *args):
        value = self.buffer.getvalue()
        if len(value) > 100 * 1024 ** 2:  # 100M
            with oss_progress('uploading') as callback:
                self.bucket.put_object(self.path, value, progress_callback=callback)
        else:
            self.bucket.put_object(self.path, value)


class NullContextWrapper:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __iter__(self):
        return self._obj.__iter__()

    def __next__(self):
        return self._obj.__next__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@contextmanager
def ignore_io_error(msg=''):
    import oss2
    try:
        yield
    except (oss2.exceptions.RequestError, oss2.exceptions.ServerError) as e:
        logger.info(str(e) + ' ' + msg)
        raise RuntimeError


@contextmanager
def mute_stderr():
    cache = sys.stderr
    sys.stderr = StringIO()
    try:
        yield None
    finally:
        sys.stderr = cache
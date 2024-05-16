# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        #self.g2net_nor = g2net_nor

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results['image_file']
        if isinstance(image_file, (list, tuple)):
            imgs = []
            for image in image_file:
                img_bytes = self.file_client.get(image)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    channel_order=self.channel_order)
                if self.to_float32:
                    img = img.astype(np.float32)
                if img is None:
                    raise ValueError(f'Fail to read {image}')
                imgs.append(img)
            results['img'] = imgs
        else:
            #print('img_file:',image_file)
            filetype = image_file.split('/')[-1].split('.')[-1]
            if filetype == 'tiff':
                backend = 'tifffile'
            else:
                backend = None
            img_bytes = self.file_client.get(image_file)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                channel_order=self.channel_order,
                backend=backend)
            if self.to_float32:
                img = img.astype(np.float32)
            if img is None:
                raise ValueError(f'Fail to read {image_file}')
            if filetype == 'tiff':
                img = img.transpose(1,2,0)
            results['img'] = img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
    
@PIPELINES.register_module()
class attention_heatmap_loading:
    """
    Load heatmap image(s) from file.

    Required key: "heatmap_file" (path to the heatmap image file).

    Added key: "heatmap" (loaded heatmap image).

    Args:
        to_float32 (bool): Whether to convert the loaded heatmap to a float32 numpy array.
                           Defaults to False.
        file_client_args (dict): Arguments to instantiate a FileClient.
                                 See :class:`mmcv.fileio.FileClient` for details.
                                 Defaults to ``dict(backend='disk')``.
    """
    '''
    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        #print('seg_prefix:',results.get('seg_prefix', None))
        filename = results.get('attention_heatmap_prefix', None)
        img_bytes = self.file_client.get(filename)
        attention_heatmap = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        results['attention_heatmap'] = attention_heatmap
        
        return results
    '''

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        #self.g2net_nor = g2net_nor

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        attention_heatmap_file = results['attention_heatmap_file']
        if isinstance(attention_heatmap_file, (list, tuple)):
            imgs = []
            for image in attention_heatmap_file:
                img_bytes = self.file_client.get(image)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    channel_order=self.channel_order)
                if self.to_float32:
                    img = img.astype(np.float32)
                if img is None:
                    raise ValueError(f'Fail to read {image}')
                imgs.append(img)
            results['attention_heatmap'] = imgs
        else:
            filetype = attention_heatmap_file.split('/')[-1].split('.')[-1]
            if filetype == 'tiff':
                backend = 'tifffile'
            else:
                backend = None
            img_bytes = self.file_client.get(attention_heatmap_file)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                channel_order=self.channel_order,
                backend=backend)
            if self.to_float32:
                img = img.astype(np.float32)
            if img is None:
                raise ValueError(f'Fail to read {attention_heatmap_file}')
            if filetype == 'tiff':
                img = img.transpose(1,2,0)
            results['attention_heatmap'] = img
            #cv2.imwrite('/data2/serie/ex.png', img)
            #print("heatmap loading", img.shape)
        return results

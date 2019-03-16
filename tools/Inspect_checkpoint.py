# from tensorflow.python.tools import inspect_checkpoint as chkp
#
#
# # path = "/root/lanenet/code/lanenet-lane-detection/model/mobilenet/mobilenet_v2_1.0_224.ckpt"
# path = "/root/lanenet/code/lanenet-lane-detection/model/tusimple_lanenet/tusimple_lanenet_mobile_2019-03-11-21-17-42.ckpt-0"
#
# # print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True)



# print only tensor v1 in checkpoint file
# chkp.print_tensors_in_checkpoint_file(path, tensor_name='v1', all_tensors=False)





s = 'lanenet_model/inference/encode/MobilenetV2/expanded_conv_15/expand/weights:0'.split(':')[0]


# i = s.find('MobilenetV2')
i = s.find('BatchNorm')
# t = s[i:]
print(i)
# print(t)

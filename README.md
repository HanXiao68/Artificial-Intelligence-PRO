# search_image_by_image
## 图像压缩，编解码，DNN
把图像用靠谱的方法进行有损压缩，压缩结果里面的噪声会大大减少；有损压缩会尽量留住图像最重要的信息，把噪声遗忘舍弃

===================================================================================

<input_image,784D>--->[ ** ENCODE-NN **] ----><10D>---->[ ** DECODE-NN **]---><ouput_image,784D>

===================================================================================

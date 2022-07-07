min(DALL·E)
===========

|Colab|   |Replicate|   |Discord|

This is a fast, minimal port of Boris Dayma’s `DALL·E
Mega <https://github.com/borisdayma/dalle-mini>`__. It has been stripped
down for inference and converted to PyTorch. The only third party
dependencies are numpy, requests, pillow and torch.

To generate a 4x4 grid of DALL·E Mega images it takes: - 89 sec with a
T4 in Colab - 48 sec with a P100 in Colab - 14 sec with an A100 on
Replicate

The flax model and code for converting it to torch can be found
`here <https://github.com/kuprel/min-dalle-flax>`__.

Install
-------

.. code:: bash

   $ pip install min-dalle

Usage
-----

Load the model parameters once and reuse the model to generate multiple
images.

.. code:: python

   from min_dalle import MinDalle

   model = MinDalle(
       is_mega=True, 
       is_reusable=True,
       models_root='./pretrained'
   )

The required models will be downloaded to ``models_root`` if they are
not already there. Once everything has finished initializing, call
``generate_image`` with some text as many times as you want. Use a
positive ``seed`` for reproducible results. Higher values for
``log2_supercondition_factor`` result in better agreement with the text
but a narrower variety of generated images. Every image token is sampled
from the top-:math:`k` most probable tokens.

.. code:: python

   image = model.generate_image(
       text='Nuclear explosion broccoli',
       seed=-1,
       grid_size=4,
       log2_k=6,
       log2_supercondition_factor=5,
       is_verbose=False
   )

   display(image)

Interactive
~~~~~~~~~~~

If the model is being used interactively (e.g. in a notebook)
``generate_image_stream`` can be used to generate a stream of images as
the model is decoding. The detokenizer adds a slight delay for each
image. Setting ``log2_mid_count`` to 3 results in a total of
``2 ** 3 = 8`` generated images. The only valid values for
``log2_mid_count`` are 0, 1, 2, 3, and 4. This is implemented in the
colab.

.. code:: python

   image_stream = model.generate_image_stream(
       text='Dali painting of WALL·E',
       seed=-1,
       grid_size=3,
       log2_mid_count=3,
       log2_k=6,
       log2_supercondition_factor=3,
       is_verbose=False
   )

   for image in image_stream:
       display(image)

Command Line
~~~~~~~~~~~~

Use ``image_from_text.py`` to generate images from the command line.

.. code:: bash

   $ python image_from_text.py --text='artificial intelligence' --no-mega

.. |Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb
.. |Replicate| image:: https://replicate.com/kuprel/min-dalle/badge
   :target: https://replicate.com/kuprel/min-dalle
.. |Discord| image:: https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white
   :target: https://discord.com/channels/823813159592001537/912729332311556136

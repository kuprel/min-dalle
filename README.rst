min(DALL·E)
===========

|Open In Colab|   |Replicate|   |Join us on Discord|

This is a fast, minimal implementation of Boris Dayma’s `DALL·E
Mega <https://github.com/borisdayma/dalle-mini>`__. It has been stripped
down for inference and converted to PyTorch. The only third party
dependencies are numpy, requests, pillow and torch.

To generate a 4x4 grid of DALL·E Mega images it takes: - 89 sec with a
T4 in Colab - 48 sec with a P100 in Colab - 14 sec with an A100 on
Replicate - TBD with an H100 (@NVIDIA?)

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

   model = MinDalle(is_mega=True, models_root='./pretrained')

The required models will be downloaded to ``models_root`` if they are
not already there. Once everything has finished initializing, call
``generate_image`` with some text and a seed as many times as you want.

.. code:: python

   text = 'Dali painting of WALL·E'
   image = model.generate_image(text, seed=0, grid_size=4)
   display(image)

.. code:: python

   text = 'Rusty Iron Man suit found abandoned in the woods being reclaimed by nature'
   image = model.generate_image(text, seed=0, grid_size=3)
   display(image)

.. code:: python

   text = 'court sketch of godzilla on trial'
   image = model.generate_image(text, seed=6, grid_size=3)
   display(image)

.. code:: python

   text = 'a funeral at Whole Foods'
   image = model.generate_image(text, seed=10, grid_size=3)
   display(image)

.. code:: python

   text = 'Jesus turning water into wine on Americas Got Talent'
   image = model.generate_image(text, seed=2, grid_size=3)
   display(image)

.. code:: python

   text = 'cctv footage of Yoda robbing a liquor store'
   image = model.generate_image(text, seed=0, grid_size=3)
   display(image)

Command Line
~~~~~~~~~~~~

Use ``image_from_text.py`` to generate images from the command line.

.. code:: bash

   $ python image_from_text.py --text='artificial intelligence' --no-mega --seed=7

.. code:: bash

   $ python image_from_text.py --text='trail cam footage of gollum eating watermelon' --mega --seed=1 --grid-size=3

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb
.. |Replicate| image:: https://replicate.com/kuprel/min-dalle/badge
   :target: https://replicate.com/kuprel/min-dalle
.. |Join us on Discord| image:: https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white
   :target: https://discord.gg/xBPBXfcFHd

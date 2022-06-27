# min-dalle


```
python3 image_from_text_flax.py \
  --dalle_bart_path='./pretrained/dalle_bart_mega' \
  --vqgan_path='./pretrained/vqgan' \
  --image_path='./generated/avacado_armchair_flax.png' \
  --seed=4 \
  --text='a comfy chair that looks like an avocado'
```
![Avocado Armchair](examples/avocado_armchair.png)


```
python3 image_from_text_flax.py \
  --dalle_path='./pretrained/dalle-mega' \
  --seed=100 \
  --image_path='./generated/godzilla_trial.png' \
  --text='court sketch of godzilla on trial'
```

![Godzilla Trial](examples/godzilla_trial.png)
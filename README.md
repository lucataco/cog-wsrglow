# zkx06111/WSRGlow Cog model

This is an updated version of the Replicate cog mode [zkx06111/WSRGlow](https://replicate.com/zkx06111/wsrglow). [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog build -t wsrglow

Then, you can run predictions:

    cog predict -i input=@demo.wav

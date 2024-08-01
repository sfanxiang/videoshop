# Videoshop

[[Project]](https://videoshop-editing.github.io/)
[[Paper]](https://arxiv.org/abs/2403.14617)
[[Supplementary]](https://videoshop-editing.github.io/static/supplementary)

Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion.

## Code Usage

Make sure the input video has 14 frames. Run:
```sh
python3 run.py --video <INPUT VIDEO.mp4> --image <INPUT IMAGE.png> --output <OUTPUT VIDEO.mp4>
```

You can find more control knobs by invoking the script with `--help`:
```sh
python3 run.py --help
```

## Citation

```
@misc{fan2024videoshop,
      title={Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion}, 
      author={Xiang Fan and Anand Bhattad and Ranjay Krishna},
      year={2024},
      eprint={2403.14617},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

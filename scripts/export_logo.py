"""Export the generated logo at display sizes, preserving its alpha channel."""
from pathlib import Path

from PIL import Image, ImageOps


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / 'foliorecall/assets'
    assets.mkdir(exist_ok=True)
    for source, output, size in [('foliorecall-wordmark.png', 'wordmark.png', (1024, 200)),
                                  ('foliorecall-lettermark.png', 'logo.png', (256, 256))]:
        with Image.open(root / 'doc/assets' / source) as image:
            rgba = image.convert('RGBA')
            # Ignore nearly transparent generation noise when locating the artwork.
            bounds = rgba.getchannel('A').point(lambda a: 255 if a > 32 else 0).getbbox()
            cropped = rgba.crop(bounds)
            # Transparent padding and resizing only; lettering and colors are unchanged.
            cropped = ImageOps.expand(cropped, border=round(cropped.height * .06))
            fitted = ImageOps.contain(cropped, size, Image.Resampling.LANCZOS)
            canvas = Image.new('RGBA', size)
            canvas.alpha_composite(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
            canvas.save(assets / output, optimize=True)
            print(output, size, (assets / output).stat().st_size)


if __name__ == '__main__':
    main()

# FolioRecall 标识

完整字标由项目名称 FolioRecall 变形构成：F 的上横融入翻页折角，R 的圆框和斜笔组成放大镜，保留竖笔以维持字母辨识度。首字母 F 单独用于浏览器图标。采用深绿与灰绿，图形保留透明底；页寻使用界面文字显示。

资源由 OpenAI 内置 image_gen 生成，未使用上游项目标识。原图为 `foliorecall-wordmark.png` 和 `foliorecall-lettermark.png`；前端与README使用 `foliorecall/assets/wordmark.png`，浏览器图标使用同目录 `logo.png`。附带发布包的 `branding` 目录提供原图和显示尺寸。

运行 `python -m scripts.export_logo` 可从原图重新导出。脚本只按可见范围裁边、增加透明留白和缩放，保持字形、颜色及透明通道；没有额外生成调用。先前页面与放大镜方案未采用。

## 完整字标生成提示

```text
Use case: logo-brand. Design an original, thoughtful CUSTOM LETTERING LOGO for the open-source document Agent project FolioRecall. Primary request: transform the project name itself into the logo, rather than putting ordinary type beside a document/search icon. Text must read exactly FolioRecall, spelled F o l i o R e c a l l, with capital F and R and all other letters lowercase. One horizontal wordmark only. Typography is the artwork: a cohesive geometric humanist logotype drawn as custom shapes, refined editorial character, confident medium-bold strokes, generous counters, clean small-size readability. The initial F is constructed with a subtly folded folio/page corner integrated into its upper bar; the capital R has a gracefully returning diagonal leg and an open circular counter suggesting retrieval and recall. Make these modifications feel like parts of the same letter system, not icons pasted over letters. The letters may share an elegant continuous baseline rhythm. Restrict conceptual transformations to F and R, keep the rest easily readable. Palette: solid dark forest teal #176A58 for Folio and most of the wordmark, muted sage #91B5A3 only on the folded part or returning R stroke. Transparent background with real alpha. Wide 3:1 canvas, wordmark centered and occupying about 88% of width with comfortable margins. Flat vector-like finish. No independent document icon, no detached magnifying glass, no badge, no caption, no Chinese subtitle, no slogan, no additional logo variations, no presentation sheet, no shadows, no texture, no gradients. This is a finished distinctive identity for a minimalist software app and GitHub README, not a mockup.
```

## R 放大镜修订提示

以初版字标为输入，按用户要求将 R 改为放大镜。当前 `foliorecall-wordmark.png` 是此轮修订结果；首字母图标沿用初版 F。

```text
Edit the supplied FolioRecall custom wordmark. Preserve the folded F and every lowercase letter exactly in design, typography, position, spelling and dark teal/sage palette. Change ONLY the capital R into an unmistakable magnifying-glass letterform, while keeping it clearly readable as R. Build its upper bowl as a truly circular magnifying-glass frame, with transparent empty lens, joined to the existing vertical left stem; use the diagonal R leg as a straight clean magnifier handle descending toward the lower right. Remove the current flowing curled ribbon leg; the new sage handle should be geometric, compact and match the thickness and rounded finishing of the rest of the type. The ring and upright are dark forest teal #176A58, handle muted sage #91B5A3. Do not add a separate magnifying-glass icon next to the text: the R itself IS the magnifier. Preserve clear separation from adjacent o and e. Keep the name spelled exactly FolioRecall, capital F and R. Preserve the minimalist flat, crisp, transparent-background logo, one horizontal wordmark, no caption, no decorative elements, no shadow, no gradients, no background fill. Maintain the same wide composition.
```

## 首字母图标生成提示

输入为初版完整字标。

```text
Use case: logo-brand. This image is the approved design source for an asset extraction. Extract ONLY the custom initial capital F from the FolioRecall wordmark: preserve its exact folded sage corner, dark teal body, rounded descending stem and curved horizontal bars. Remove every other letter. Do not redesign the F or add any symbols. Place that single F centered on a square genuinely transparent canvas with comfortable 12 percent padding, suitable as the matching browser favicon/app avatar. Preserve original palette and crisp contours. No other text, no shadow, no enclosing square, no mockup.
```

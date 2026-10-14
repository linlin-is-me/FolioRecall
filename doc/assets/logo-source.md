# FolioRecall 标识

完整字标由项目名称 FolioRecall 变形构成：F 的上横融入翻页折角，R 左侧的竖立书本构成竖笔，右侧放大镜构成圆部与斜腿；末尾两个 l 保留清晰竖笔，以底部页边和书脊连接成一本书。首字母 F 单独用于浏览器图标。采用深绿与灰绿，图形保留透明底；页寻使用界面文字显示。完整项目名同时保留在README标题、图片替代文字和浏览器标题中。

资源由 OpenAI 内置 image_gen 生成，未使用上游项目标识。原图为 `foliorecall-wordmark.png` 和 `foliorecall-lettermark.png`；前端与README使用 `foliorecall/assets/wordmark.png`，浏览器图标使用同目录 `logo.png`。附带发布包的 `branding` 目录提供原图和显示尺寸。

运行 `python -m scripts.export_logo` 可从原图重新导出。脚本只按可见范围裁边、增加透明留白和缩放，保持字形、颜色及透明通道；没有额外生成调用。先前页面与放大镜方案未采用。

## 完整字标生成提示

```text
Use case: logo-brand. Design an original, thoughtful CUSTOM LETTERING LOGO for the open-source document Agent project FolioRecall. Primary request: transform the project name itself into the logo, rather than putting ordinary type beside a document/search icon. Text must read exactly FolioRecall, spelled F o l i o R e c a l l, with capital F and R and all other letters lowercase. One horizontal wordmark only. Typography is the artwork: a cohesive geometric humanist logotype drawn as custom shapes, refined editorial character, confident medium-bold strokes, generous counters, clean small-size readability. The initial F is constructed with a subtly folded folio/page corner integrated into its upper bar; the capital R has a gracefully returning diagonal leg and an open circular counter suggesting retrieval and recall. Make these modifications feel like parts of the same letter system, not icons pasted over letters. The letters may share an elegant continuous baseline rhythm. Restrict conceptual transformations to F and R, keep the rest easily readable. Palette: solid dark forest teal #176A58 for Folio and most of the wordmark, muted sage #91B5A3 only on the folded part or returning R stroke. Transparent background with real alpha. Wide 3:1 canvas, wordmark centered and occupying about 88% of width with comfortable margins. Flat vector-like finish. No independent document icon, no detached magnifying glass, no badge, no caption, no Chinese subtitle, no slogan, no additional logo variations, no presentation sheet, no shadows, no texture, no gradients. This is a finished distinctive identity for a minimalist software app and GitHub README, not a mockup.
```

## R 放大镜修订提示

以初版字标为输入，第一次修订仍保留R竖笔。用户随后要求放大镜直接可辨，最终采用下方的完整图形替换；首字母图标沿用初版F。

```text
Edit the supplied FolioRecall custom wordmark. Preserve the folded F and every lowercase letter exactly in design, typography, position, spelling and dark teal/sage palette. Change ONLY the capital R into an unmistakable magnifying-glass letterform, while keeping it clearly readable as R. Build its upper bowl as a truly circular magnifying-glass frame, with transparent empty lens, joined to the existing vertical left stem; use the diagonal R leg as a straight clean magnifier handle descending toward the lower right. Remove the current flowing curled ribbon leg; the new sage handle should be geometric, compact and match the thickness and rounded finishing of the rest of the type. The ring and upright are dark forest teal #176A58, handle muted sage #91B5A3. Do not add a separate magnifying-glass icon next to the text: the R itself IS the magnifier. Preserve clear separation from adjacent o and e. Keep the name spelled exactly FolioRecall, capital F and R. Preserve the minimalist flat, crisp, transparent-background logo, one horizontal wordmark, no caption, no decorative elements, no shadow, no gradients, no background fill. Maintain the same wide composition.
```

## R 直接替换为放大镜

```text
Make ONE very specific edit to this FolioRecall logo. The user wants an immediately recognizable literal MAGNIFYING GLASS in the position currently occupied by the uppercase R. DELETE THE ENTIRE R INCLUDING ITS LEFT VERTICAL STEM. Replace it with the familiar search icon: a clean perfectly circular ring with a large transparent empty lens, and an obvious short straight thick handle extending diagonally down and right at 45 degrees OUTSIDE the ring. No vertical stem at all, no P or R skeleton, no ribbon, no typographic interpretation of R. It must instantly look like a magnifying glass, not a letter with a stylized leg. The circle's stroke should be approximately half the thickness of the neighboring lowercase letters, so the lens looks spacious and unmistakable; the handle should be solid, clearly connected to the ring's outer bottom-right edge and remain visibly separate from the next e. Use dark forest teal for the circular frame and muted sage for the handle. Size the magnifier to fit the capital-letter space with its handle aligned to the baseline. Preserve all other letters exactly: Folio [magnifying glass icon replacing R] ecall. Preserve the distinctive folded corner F, the original letter shapes, overall typography, green palette, horizontal arrangement and generous transparency. Do not keep any residual stroke of the R. Do not add extra icons elsewhere. Wide transparent canvas. Flat crisp logo, no shadows, no gradients, no new text. The purpose is a name-integrated logo in which the search symbol is unambiguous at small size.
```

## 末尾 ll 合并为书本

输入为已经替换放大镜的字标。此轮书本替换方案已由下方修订取代。

```text
Edit ONLY the final two lowercase l letters at the far right of this existing logo. Preserve the folded F, all letters before the final ll, and especially the literal magnifying glass in the R position EXACTLY as supplied. Transform the terminal ll into ONE unmistakable small OPEN BOOK integrated into the wordmark: the pair of tall vertical strokes become the upright outside edges of two open pages, separated by a clear central spine/gutter; the upper and lower page contours gently meet at that central spine. Make the two facing pages visually read as a book, with open transparent interior and one restrained inset page-edge line on each side. Keep a tall compact silhouette recalling the two lowercase l stems, but allow this final unit modest extra width so the book is unmistakable at small UI size. Use dark forest teal outlines with one small muted sage page-edge accent; match the rounded geometric family and visual stroke weight of the wordmark. This book REPLACES the final ll, do not append an additional book after ll, and do not retain two ordinary l letters beside it. Keep visual text sequence F-o-l-i-o-[magnifying glass]-e-c-a-[open book representing ll]. Same baseline and cap height. No change to magnifying glass, no extra text, no unrelated decoration, no shadow or background. Flat clean logo on genuinely transparent background, one horizontal composition.
```

## 首字母图标生成提示

输入为初版完整字标。

```text
Use case: logo-brand. This image is the approved design source for an asset extraction. Extract ONLY the custom initial capital F from the FolioRecall wordmark: preserve its exact folded sage corner, dark teal body, rounded descending stem and curved horizontal bars. Remove every other letter. Do not redesign the F or add any symbols. Place that single F centered on a square genuinely transparent canvas with comfortable 12 percent padding, suitable as the matching browser favicon/app avatar. Preserve original palette and crisp contours. No other text, no shadow, no enclosing square, no mockup.
```

## 完整 R 与清晰 ll 的最终修订

用户要求将书本放到放大镜右侧共同组成R，末尾ll恢复可辨识字母。以下为本轮编辑及补齐R竖笔的提示，使用内置image_gen；此轮左右安排已由下方修订取代。

```text
Use case: logo-brand, precise edit. Edit the supplied FolioRecall wordmark according to this exact request: R字母的放大镜部分右侧加上一本书组成完整的字母R，最后两个ll要能看出是字母l. Keep the folded F and all untouched lowercase letters, palette and minimalist geometric style. TWO changes only: (1) At the R position, combine the existing clearly recognizable magnifying glass WITH A SMALL BOOK ON ITS RIGHT into one cohesive, clearly readable CAPITAL R. The magnifying-glass circular lens and diagonal handle must remain recognizable, but restore the necessary upright structure so the whole custom glyph reads R rather than Q or an isolated search icon. Integrate a compact upright book/page detail into the RIGHT SIDE of this composite R, not somewhere else in the wordmark; its straight spine and subtle page edges should participate in the R structure. Do not add a large standalone book between R and e. Preserve clean spacing and a strong R silhouette at small size. (2) Replace the terminal open-book icon with TWO SEPARATE CLEAR LOWERCASE l LETTERS, same height as the earlier l in Folio, upright parallel stems with a distinct gap. They may have a tiny page-edge detail, but must read 'll', not an open book, m, n, or an icon. Correct full readable spelling is FolioRecall. Dark forest teal with muted sage accents, genuine transparent background, flat crisp finish, no shadows, no new words, no slogan. Only one horizontal wordmark.

Precise logo correction. Keep the supplied artwork unchanged EXCEPT complete the capital R at the center. It currently has a magnifying-glass ring, book on its RIGHT, and diagonal leg, but it is MISSING THE LEFT VERTICAL STEM. ADD a straight dark-teal vertical stem at the LEFT EDGE of the magnifier, extending from cap height down to the shared baseline, connected to the ring; this must make the compound symbol unmistakably read capital R. Keep the circular lens open and recognizable as a magnifying glass. Keep the compact book on the RIGHT of the lens, and keep the sage diagonal handle as the R leg. Do not move the book to the left. Preserve the two distinct tall lowercase l letters at the end with their subtle sage page-edge lines, exactly as in the reference. Preserve folded F, all other letters, colors, proportions and transparent background. The exact readable name is FolioRecall. No extra symbols or text. One horizontal wordmark.
```

## 左书右镜与 ll 单本书修订

按用户澄清，R左侧为竖立书本，右侧为放大镜；末尾两个l保留字母辨识度，同时用页边连接为一本书。当前资源采用此版，使用内置image_gen编辑，提示如下。

```text
Use case: logo-brand. Precisely edit the supplied FolioRecall logo. Follow this spatial construction literally. R: ONE SLIM CLOSED BOOK STANDING VERTICALLY ON THE LEFT forms the full-height LEFT STEM of the capital R. Give this left book a recognizable narrow rectangular cover, tiny top/bottom page-edge marks and one sage inset spine line. ON THE RIGHT OF THIS VERTICAL BOOK is ONE MAGNIFYING GLASS: circular transparent lens in the upper-right and straight handle descending diagonally lower-right, together forming the R bowl and leg. The magnifier touches the upper portion of the LEFT book so together they unmistakably spell capital R. REMOVE the current book-shaped protrusion and page stripes on the RIGHT of the lens. There must be NO BOOK ON THE RIGHT OF THE R; the BOOK IS ITS LEFT STEM, the MAGNIFIER IS ITS RIGHT HALF. Terminal ll: keep TWO clear tall lowercase l stems with a visible narrow central separation, equal to the l height earlier in Folio. Together these two l strokes must also resemble ONE upright slightly open book, not two separate closed books. Add restrained thin page contours and a shallow connecting bottom cover/central gutter to suggest two facing halves of ONE book. The two dominant vertical strokes must remain readable as ll first, with the single-book reading second; do not replace ll with a squat generic open-book icon. Preserve folded F, remaining letters, original forest-teal/sage palette, transparent background, clean geometric typography and horizontal wordmark. Correct readable name FolioRecall. Flat graphic, no shadows, no gradients, no captions or additional icons.
```

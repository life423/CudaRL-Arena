# Car sprites

`cars_topdown.png` is the original PNG supplied by the project owner on
October 26, 2026, generated with ChatGPT for this project. It is 1774 × 887
pixels, with four columns and two rows, facing up: red, orange, yellow, green,
cyan, blue, purple, pink.

The source PNG is preserved byte-for-byte. The renderer derives the cell
boundaries from the actual dimensions and finds each car's visible bounds with
an alpha threshold of 0.25, plus two pixels of padding. This excludes faint stray
pixels that otherwise make the cars appear incorrectly scaled. All eight crops
are 216 × 395 pixels for this sheet.

Godot imports the sheet losslessly with alpha-border correction and mipmaps.
Each atlas region is drawn into the existing 2 × 4 metre footprint, rotated by
`PI/2 - heading` for screen coordinates. Rear brake lamps, car numbers and the
inspection ring are drawn separately. This asset change does not change the
physics or policy.

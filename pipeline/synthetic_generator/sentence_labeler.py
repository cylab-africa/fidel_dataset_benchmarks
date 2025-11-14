import textwrap
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from tqdm import tqdm
def txt_to_images_with_csv(
    txt_path,
    csv_path,
    output_prefix="image",
    char_wrap=50,
    dpi=300,
    font_path=None
):
    """
    Reads each line of txt_path. For lines that have >5 words:
    1) Wrap them (char_wrap)
    2) Create a figure sized to fit the wrapped text
    3) Save as a PNG (one image per line)
    4) Write (filename, full_line) to CSV
    """
    # 1. Read all lines from the text file
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = set(lines)
    # 2. Prepare the CSV writer
    with open(csv_path, 'a+', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write a header row (optional)
        writer.writerow(["filename", "label"])

        # 3. (Optional) register a custom font (e.g., for Amharic/Ethiopic)
        if font_path and os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False

        # 4. Loop each line: if >5 words, convert to image & log in CSV
        min_words = 6
        max_words_per_line = 20
        pbar = tqdm(total=len(lines), desc="Creating files")
        for i, line in enumerate(lines):
            if True:
                words = line.split()
                # Strip and split
                text_line = line.strip()
                words = text_line.split()

                # Skip lines that don't meet the min word requirement
                if len(words) < min_words:
                    continue

                # 5) Word-based wrap:
                #    Split 'words' into sub-lines, each up to 'max_words_per_line'
                sub_lines = []
                for start in range(0, len(words), max_words_per_line):
                    chunk = words[start: start + max_words_per_line]
                    sub_line = " ".join(chunk)
                    sub_lines.append(sub_line)

                # If sub_lines is empty, skip
                if not sub_lines:
                    continue

                if len(words) > min_words:
                    text_line = line.strip()  # remove trailing spaces/newlines
                    # You can add a suffix if you want, e.g. text_line += "::"

                    # Wrap the line
                    wrapped_lines = textwrap.wrap(text_line, width=char_wrap)
                    n_wrapped = len(wrapped_lines)
                    if n_wrapped == 0:
                        continue

                    # Dynamically size figure height
                    line_height_in = 0.4
                    top_bottom_margin = 0.4
                    fig_height = line_height_in + top_bottom_margin
                    fig_width = 10.0

                    # Create figure

                    for j, sub_line in enumerate(sub_lines):
                        out_filename = f"{output_prefix}_{i}_{j}.png"
                        if j == len(sub_lines) - 1:
                            sub_line = f"{sub_line}::"

                        if not ( os.path.exists(f"data\{out_filename}")):

                            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                            ax.axis('off')


                            # We’ll use data coordinates: x from [0..1], y from [0..fig_height]
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, fig_height)

                            current_y = fig_height - 0.3  # top margin

                            ax.text(
                                0.05,
                                current_y,
                                sub_line,
                                fontsize=14,
                                ha='left',
                                va='top'
                            )


                            # Construct an output filename
                            out_filename = f"{output_prefix}_{i}_{j}.png"
                            plt.savefig(f"data\{out_filename}", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
                            plt.close(fig)

                        # print(f"Created: {out_filename}")

                        # 5. Write the (filename, label) pair to CSV
                        writer.writerow([out_filename, sub_line])
            pbar.update(1)
        pbar.close()
def main():
    txt_file = r"C:\Users\gudab\Downloads\am.txt\combined_amharic_sentences.txt"
    csv_file = "labels\output_labels.csv"

    # If you have an Ethiopic-capable font, specify its path:
    font_path = r"static\NotoSansEthiopic-Black.ttf"

    txt_to_images_with_csv(
        txt_file,
        csv_file,
        output_prefix="image",
        char_wrap=50,
        dpi=300,
        font_path=font_path
    )
    print(f"CSV created: {csv_file}")

if __name__ == "__main__":
    main()

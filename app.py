import streamlit as st
import fitz  # PyMuPDF
import pikepdf
import tempfile
import os
import json
import hashlib
import zipfile
from pathlib import Path
from collections import Counter

from PIL import Image
import cv2
import numpy as np
import pandas as pd


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Engineering Drawing Extractor",
    page_icon="📐",
    layout="wide"
)

st.title("Engineering Drawing Extractor")

st.write(
    "Upload an engineering PDF or image and extract "
    "native text, coordinates, vectors, comments, images, "
    "metadata and high-resolution renders."
)


# ============================================================
# HELPERS
# ============================================================

def json_safe(value):

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, fitz.Rect):
        return [
            value.x0,
            value.y0,
            value.x1,
            value.y1
        ]

    if isinstance(value, fitz.Point):
        return [
            value.x,
            value.y
        ]

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    try:
        return str(value)
    except Exception:
        return None


def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def sha256_file(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text(page):

    data = {
        "plain_text": "",
        "words": [],
        "blocks": [],
        "lines": [],
        "spans": [],
        "tiny_text": [],
        "rotated_text": []
    }

    data["plain_text"] = page.get_text(
        "text"
    )

    # --------------------------------------------------------
    # Words + coordinates
    # --------------------------------------------------------

    words = page.get_text(
        "words",
        sort=False
    )

    for word in words:

        if len(word) < 8:
            continue

        (
            x0,
            y0,
            x1,
            y1,
            text,
            block_no,
            line_no,
            word_no
        ) = word[:8]

        data["words"].append({
            "text": text,
            "bbox": [
                x0,
                y0,
                x1,
                y1
            ],
            "block_number":
                block_no,
            "line_number":
                line_no,
            "word_number":
                word_no
        })

    # --------------------------------------------------------
    # Rich spans
    # --------------------------------------------------------

    raw = page.get_text(
        "dict",
        sort=False
    )

    for block_index, block in enumerate(
        raw.get("blocks", [])
    ):

        data["blocks"].append({
            "block_index":
                block_index,

            "type":
                block.get("type"),

            "bbox":
                json_safe(
                    block.get("bbox")
                )
        })

        if "lines" not in block:
            continue

        for line_index, line in enumerate(
            block["lines"]
        ):

            direction = line.get(
                "dir",
                (1.0, 0.0)
            )

            data["lines"].append({
                "block_index":
                    block_index,

                "line_index":
                    line_index,

                "bbox":
                    json_safe(
                        line.get("bbox")
                    ),

                "direction":
                    json_safe(direction),

                "writing_mode":
                    line.get("wmode")
            })

            for span_index, span in enumerate(
                line.get("spans", [])
            ):

                record = {
                    "text":
                        span.get(
                            "text",
                            ""
                        ),

                    "bbox":
                        json_safe(
                            span.get("bbox")
                        ),

                    "origin":
                        json_safe(
                            span.get("origin")
                        ),

                    "font":
                        span.get("font"),

                    "font_size":
                        span.get("size"),

                    "color":
                        span.get("color"),

                    "alpha":
                        span.get("alpha"),

                    "flags":
                        span.get("flags"),

                    "char_flags":
                        span.get(
                            "char_flags"
                        ),

                    "ascender":
                        span.get(
                            "ascender"
                        ),

                    "descender":
                        span.get(
                            "descender"
                        ),

                    "direction":
                        json_safe(
                            direction
                        ),

                    "block_index":
                        block_index,

                    "line_index":
                        line_index,

                    "span_index":
                        span_index
                }

                data["spans"].append(
                    record
                )

                font_size = span.get(
                    "size",
                    100
                )

                if font_size < 4:

                    data[
                        "tiny_text"
                    ].append(
                        record
                    )

                if tuple(direction) != (
                    1.0,
                    0.0
                ):

                    data[
                        "rotated_text"
                    ].append(
                        record
                    )

    return data


# ============================================================
# PDF VECTOR EXTRACTION
# ============================================================

def extract_vectors(page):

    result = []

    drawings = page.get_drawings()

    for index, drawing in enumerate(
        drawings
    ):

        record = {
            "index":
                index,

            "type":
                drawing.get(
                    "type"
                ),

            "bbox":
                json_safe(
                    drawing.get(
                        "rect"
                    )
                ),

            "stroke_color":
                json_safe(
                    drawing.get(
                        "color"
                    )
                ),

            "fill_color":
                json_safe(
                    drawing.get(
                        "fill"
                    )
                ),

            "line_width":
                drawing.get(
                    "width"
                ),

            "line_cap":
                json_safe(
                    drawing.get(
                        "lineCap"
                    )
                ),

            "line_join":
                drawing.get(
                    "lineJoin"
                ),

            "dashes":
                drawing.get(
                    "dashes"
                ),

            "close_path":
                drawing.get(
                    "closePath"
                ),

            "stroke_opacity":
                drawing.get(
                    "stroke_opacity"
                ),

            "fill_opacity":
                drawing.get(
                    "fill_opacity"
                ),

            "layer":
                drawing.get(
                    "layer"
                ),

            "sequence_number":
                drawing.get(
                    "seqno"
                ),

            "items": []
        }

        for item in drawing.get(
            "items",
            []
        ):

            record["items"].append({
                "command":
                    item[0],

                "data":
                    json_safe(
                        list(
                            item[1:]
                        )
                    )
            })

        result.append(record)

    return result


# ============================================================
# PDF ANNOTATIONS
# ============================================================

def extract_annotations(page):

    result = []

    annot = page.first_annot

    while annot:

        info = annot.info or {}

        record = {
            "xref":
                annot.xref,

            "type":
                json_safe(
                    annot.type
                ),

            "bbox":
                json_safe(
                    annot.rect
                ),

            "flags":
                annot.flags,

            "colors":
                json_safe(
                    annot.colors
                ),

            "opacity":
                annot.opacity,

            "blend_mode":
                annot.blendmode,

            "content":
                info.get(
                    "content"
                ),

            "title":
                info.get(
                    "title"
                ),

            "subject":
                info.get(
                    "subject"
                ),

            "creation_date":
                info.get(
                    "creationDate"
                ),

            "modified_date":
                info.get(
                    "modDate"
                )
        }

        try:

            record["vertices"] = (
                json_safe(
                    annot.vertices
                )
            )

        except Exception:

            record["vertices"] = None

        result.append(record)

        annot = annot.next

    return result


# ============================================================
# LINKS
# ============================================================

def extract_links(page):

    return [
        json_safe(link)
        for link in page.get_links()
    ]


# ============================================================
# EMBEDDED IMAGES
# ============================================================

def extract_images(
    doc,
    page,
    page_number,
    output_dir
):

    image_dir = os.path.join(
        output_dir,
        "embedded_images"
    )

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    result = []

    seen = set()

    for image_index, image in enumerate(
        page.get_images(
            full=True
        )
    ):

        xref = image[0]

        if xref in seen:
            continue

        seen.add(xref)

        record = {
            "image_index":
                image_index,

            "xref":
                xref,

            "descriptor":
                json_safe(
                    image
                )
        }

        try:

            extracted = (
                doc.extract_image(
                    xref
                )
            )

            ext = extracted.get(
                "ext",
                "bin"
            )

            filename = (
                f"page_{page_number + 1}_"
                f"xref_{xref}.{ext}"
            )

            path = os.path.join(
                image_dir,
                filename
            )

            with open(
                path,
                "wb"
            ) as f:

                f.write(
                    extracted[
                        "image"
                    ]
                )

            record.update({
                "filename":
                    filename,

                "width":
                    extracted.get(
                        "width"
                    ),

                "height":
                    extracted.get(
                        "height"
                    ),

                "colorspace":
                    extracted.get(
                        "colorspace"
                    ),

                "bpc":
                    extracted.get(
                        "bpc"
                    )
            })

            try:

                rects = (
                    page.get_image_rects(
                        xref
                    )
                )

                record[
                    "locations"
                ] = [
                    json_safe(r)
                    for r in rects
                ]

            except Exception as exc:

                record[
                    "location_error"
                ] = str(exc)

        except Exception as exc:

            record["error"] = (
                str(exc)
            )

        result.append(record)

    return result


# ============================================================
# PAGE RENDER
# ============================================================

def render_page(
    page,
    page_number,
    output_dir,
    zoom=4
):

    render_dir = os.path.join(
        output_dir,
        "renders"
    )

    os.makedirs(
        render_dir,
        exist_ok=True
    )

    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    filename = (
        f"page_{page_number + 1}_"
        f"{zoom}x.png"
    )

    path = os.path.join(
        render_dir,
        filename
    )

    pix.save(path)

    return {
        "filename":
            filename,

        "path":
            path,

        "width":
            pix.width,

        "height":
            pix.height,

        "zoom":
            zoom
    }


# ============================================================
# RAW PDF OBJECTS
# ============================================================

def extract_raw_objects(
    pdf_path,
    output_dir
):

    raw_dir = os.path.join(
        output_dir,
        "raw_objects"
    )

    os.makedirs(
        raw_dir,
        exist_ok=True
    )

    result = []

    try:

        pdf = pikepdf.Pdf.open(
            pdf_path
        )

        for index, obj in enumerate(
            pdf.objects,
            start=1
        ):

            record = {
                "object_number":
                    index,

                "type":
                    type(
                        obj
                    ).__name__
            }

            try:

                record[
                    "representation"
                ] = str(obj)

            except Exception:

                record[
                    "representation"
                ] = None

            try:

                if isinstance(
                    obj,
                    pikepdf.Stream
                ):

                    data = (
                        obj.read_raw_bytes()
                    )

                    filename = (
                        f"object_{index}.bin"
                    )

                    path = os.path.join(
                        raw_dir,
                        filename
                    )

                    with open(
                        path,
                        "wb"
                    ) as f:

                        f.write(data)

                    record[
                        "raw_stream_file"
                    ] = filename

                    record[
                        "stream_size"
                    ] = len(data)

            except Exception as exc:

                record[
                    "stream_error"
                ] = str(exc)

            result.append(
                record
            )

        pdf.close()

    except Exception as exc:

        return {
            "error":
                str(exc)
        }

    return {
        "objects":
            result
    }


# ============================================================
# PDF STATISTICS
# ============================================================

def build_statistics(
    text_data,
    vectors,
    annotations,
    images
):

    fonts = Counter()
    sizes = Counter()
    vector_colors = Counter()

    for span in text_data[
        "spans"
    ]:

        fonts[
            str(
                span.get(
                    "font"
                )
            )
        ] += 1

        size = span.get(
            "font_size"
        )

        if size is not None:

            sizes[
                str(
                    round(
                        float(size),
                        2
                    )
                )
            ] += 1

    for vector in vectors:

        vector_colors[
            str(
                vector.get(
                    "stroke_color"
                )
            )
        ] += 1

    return {
        "word_count":
            len(
                text_data[
                    "words"
                ]
            ),

        "span_count":
            len(
                text_data[
                    "spans"
                ]
            ),

        "tiny_text_count":
            len(
                text_data[
                    "tiny_text"
                ]
            ),

        "rotated_text_count":
            len(
                text_data[
                    "rotated_text"
                ]
            ),

        "vector_count":
            len(
                vectors
            ),

        "annotation_count":
            len(
                annotations
            ),

        "embedded_image_count":
            len(
                images
            ),

        "fonts":
            dict(fonts),

        "font_sizes":
            dict(sizes),

        "vector_colors":
            dict(
                vector_colors
            )
    }


# ============================================================
# COMPLETE PDF EXTRACTOR
# ============================================================

def extract_pdf(
    pdf_path,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    doc = fitz.open(
        pdf_path
    )

    master = {
        "source_file":
            os.path.basename(
                pdf_path
            ),

        "sha256":
            sha256_file(
                pdf_path
            ),

        "metadata":
            json_safe(
                doc.metadata
            ),

        "page_count":
            len(doc),

        "pages": []
    }

    all_text = []
    all_vectors = []
    all_annotations = []
    all_images = []

    for page_number in range(
        len(doc)
    ):

        page = doc[
            page_number
        ]

        metadata = {
            "page_number":
                page_number + 1,

            "rotation":
                page.rotation,

            "page_rect":
                json_safe(
                    page.rect
                ),

            "mediabox":
                json_safe(
                    page.mediabox
                ),

            "cropbox":
                json_safe(
                    page.cropbox
                ),

            "bleedbox":
                json_safe(
                    page.bleedbox
                ),

            "trimbox":
                json_safe(
                    page.trimbox
                ),

            "artbox":
                json_safe(
                    page.artbox
                )
        }

        text_data = (
            extract_text(
                page
            )
        )

        vectors = (
            extract_vectors(
                page
            )
        )

        annotations = (
            extract_annotations(
                page
            )
        )

        images = (
            extract_images(
                doc,
                page,
                page_number,
                output_dir
            )
        )

        links = (
            extract_links(
                page
            )
        )

        render_4x = (
            render_page(
                page,
                page_number,
                output_dir,
                zoom=4
            )
        )

        stats = (
            build_statistics(
                text_data,
                vectors,
                annotations,
                images
            )
        )

        master[
            "pages"
        ].append({
            "metadata":
                metadata,

            "statistics":
                stats,

            "links":
                links,

            "render":
                render_4x
        })

        all_text.append({
            "page":
                page_number + 1,

            "data":
                text_data
        })

        all_vectors.append({
            "page":
                page_number + 1,

            "data":
                vectors
        })

        all_annotations.append({
            "page":
                page_number + 1,

            "data":
                annotations
        })

        all_images.append({
            "page":
                page_number + 1,

            "data":
                images
        })

    doc.close()

    # --------------------------------------------------------
    # Save separate JSON files
    # --------------------------------------------------------

    save_json(
        master,
        os.path.join(
            output_dir,
            "page_metadata.json"
        )
    )

    save_json(
        all_text,
        os.path.join(
            output_dir,
            "text.json"
        )
    )

    save_json(
        all_vectors,
        os.path.join(
            output_dir,
            "vectors.json"
        )
    )

    save_json(
        all_annotations,
        os.path.join(
            output_dir,
            "annotations.json"
        )
    )

    save_json(
        all_images,
        os.path.join(
            output_dir,
            "images.json"
        )
    )

    raw_objects = (
        extract_raw_objects(
            pdf_path,
            output_dir
        )
    )

    save_json(
        raw_objects,
        os.path.join(
            output_dir,
            "raw_objects.json"
        )
    )

    combined = {
        "document":
            master,

        "text":
            all_text,

        "vectors":
            all_vectors,

        "annotations":
            all_annotations,

        "images":
            all_images
    }

    save_json(
        combined,
        os.path.join(
            output_dir,
            "all_objects.json"
        )
    )

    # --------------------------------------------------------
    # Plain text output
    # --------------------------------------------------------

    with open(
        os.path.join(
            output_dir,
            "all_text.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        for page_info in all_text:

            f.write(
                f"\n\n===== PAGE "
                f"{page_info['page']} =====\n\n"
            )

            f.write(
                page_info[
                    "data"
                ][
                    "plain_text"
                ]
            )

    return {
        "master":
            master,

        "text":
            all_text,

        "vectors":
            all_vectors,

        "annotations":
            all_annotations,

        "images":
            all_images,

        "output_dir":
            output_dir
    }


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_image_file(
    image_path,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Could not read image."
        )

    height, width = (
        image.shape[:2]
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Threshold version
    # --------------------------------------------------------

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    threshold_path = os.path.join(
        output_dir,
        "threshold.png"
    )

    cv2.imwrite(
        threshold_path,
        threshold
    )

    # --------------------------------------------------------
    # Edge extraction
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edges_path = os.path.join(
        output_dir,
        "edges.png"
    )

    cv2.imwrite(
        edges_path,
        edges
    )

    # --------------------------------------------------------
    # Line detection
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    line_records = []

    preview = image.copy()

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = (
                line[0]
            )

            line_records.append({
                "x1":
                    int(x1),

                "y1":
                    int(y1),

                "x2":
                    int(x2),

                "y2":
                    int(y2)
            })

            cv2.line(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                1
            )

    preview_path = os.path.join(
        output_dir,
        "detected_lines.png"
    )

    cv2.imwrite(
        preview_path,
        preview
    )

    result = {
        "source_file":
            os.path.basename(
                image_path
            ),

        "width":
            width,

        "height":
            height,

        "detected_line_count":
            len(
                line_records
            ),

        "lines":
            line_records,

        "threshold_file":
            threshold_path,

        "edges_file":
            edges_path,

        "line_preview_file":
            preview_path
    }

    save_json(
        result,
        os.path.join(
            output_dir,
            "image_extraction.json"
        )
    )

    return result


# ============================================================
# ZIP OUTPUT
# ============================================================

def create_zip(
    source_dir,
    zip_path
):

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zipf:

        for root, dirs, files in os.walk(
            source_dir
        ):

            for file in files:

                file_path = os.path.join(
                    root,
                    file
                )

                arcname = os.path.relpath(
                    file_path,
                    source_dir
                )

                zipf.write(
                    file_path,
                    arcname
                )


# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader(
    "Upload engineering drawing",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)

if uploaded_file is not None:

    st.success(
        f"Uploaded: "
        f"{uploaded_file.name}"
    )

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    temp_dir = tempfile.mkdtemp()

    input_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(
        input_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # ========================================================
    # PDF
    # ========================================================

    if suffix == ".pdf":

        st.subheader(
            "PDF Drawing"
        )

        if st.button(
            "Extract Everything"
        ):

            output_dir = os.path.join(
                temp_dir,
                "extracted"
            )

            with st.spinner(
                "Extracting PDF structure..."
            ):

                result = extract_pdf(
                    input_path,
                    output_dir
                )

            st.success(
                "Extraction completed."
            )

            master = result[
                "master"
            ]

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "Extraction Summary"
            )

            rows = []

            for page in master[
                "pages"
            ]:

                stats = page[
                    "statistics"
                ]

                rows.append({
                    "Page":
                        page[
                            "metadata"
                        ][
                            "page_number"
                        ],

                    "Words":
                        stats[
                            "word_count"
                        ],

                    "Text Spans":
                        stats[
                            "span_count"
                        ],

                    "Tiny Text":
                        stats[
                            "tiny_text_count"
                        ],

                    "Rotated Text":
                        stats[
                            "rotated_text_count"
                        ],

                    "Vectors":
                        stats[
                            "vector_count"
                        ],

                    "Annotations":
                        stats[
                            "annotation_count"
                        ],

                    "Images":
                        stats[
                            "embedded_image_count"
                        ]
                })

            st.dataframe(
                pd.DataFrame(
                    rows
                ),
                use_container_width=True
            )

            # ------------------------------------------------
            # TABS
            # ------------------------------------------------

            tabs = st.tabs([
                "Rendered Drawing",
                "Text",
                "Tiny Text",
                "Annotations",
                "Vectors",
                "Images",
                "Metadata",
                "Raw JSON"
            ])

            # ------------------------------------------------
            # RENDER
            # ------------------------------------------------

            with tabs[0]:

                for page in master[
                    "pages"
                ]:

                    render_path = page[
                        "render"
                    ][
                        "path"
                    ]

                    st.image(
                        render_path,
                        caption=(
                            f"Page "
                            f"{page['metadata']['page_number']}"
                        ),
                        use_container_width=True
                    )

            # ------------------------------------------------
            # TEXT
            # ------------------------------------------------

            with tabs[1]:

                for page in result[
                    "text"
                ]:

                    st.markdown(
                        f"### Page "
                        f"{page['page']}"
                    )

                    st.text_area(
                        "Extracted native text",
                        value=page[
                            "data"
                        ][
                            "plain_text"
                        ],
                        height=400,
                        key=(
                            f"text_"
                            f"{page['page']}"
                        )
                    )

            # ------------------------------------------------
            # TINY TEXT
            # ------------------------------------------------

            with tabs[2]:

                tiny_rows = []

                for page in result[
                    "text"
                ]:

                    for span in page[
                        "data"
                    ][
                        "tiny_text"
                    ]:

                        tiny_rows.append({
                            "Page":
                                page[
                                    "page"
                                ],

                            "Text":
                                span[
                                    "text"
                                ],

                            "Font Size":
                                span[
                                    "font_size"
                                ],

                            "BBox":
                                str(
                                    span[
                                        "bbox"
                                    ]
                                ),

                            "Color":
                                span[
                                    "color"
                                ]
                        })

                if tiny_rows:

                    st.dataframe(
                        pd.DataFrame(
                            tiny_rows
                        ),
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No tiny native text detected."
                    )

            # ------------------------------------------------
            # ANNOTATIONS
            # ------------------------------------------------

            with tabs[3]:

                annotation_rows = []

                for page in result[
                    "annotations"
                ]:

                    for ann in page[
                        "data"
                    ]:

                        annotation_rows.append({
                            "Page":
                                page[
                                    "page"
                                ],

                            "Type":
                                str(
                                    ann[
                                        "type"
                                    ]
                                ),

                            "Content":
                                ann[
                                    "content"
                                ],

                            "Title":
                                ann[
                                    "title"
                                ],

                            "Subject":
                                ann[
                                    "subject"
                                ],

                            "BBox":
                                str(
                                    ann[
                                        "bbox"
                                    ]
                                )
                        })

                if annotation_rows:

                    st.dataframe(
                        pd.DataFrame(
                            annotation_rows
                        ),
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No PDF annotations found."
                    )

            # ------------------------------------------------
            # VECTORS
            # ------------------------------------------------

            with tabs[4]:

                vector_rows = []

                for page in result[
                    "vectors"
                ]:

                    for vec in page[
                        "data"
                    ][:500]:

                        vector_rows.append({
                            "Page":
                                page[
                                    "page"
                                ],

                            "Index":
                                vec[
                                    "index"
                                ],

                            "Type":
                                vec[
                                    "type"
                                ],

                            "Stroke":
                                str(
                                    vec[
                                        "stroke_color"
                                    ]
                                ),

                            "Fill":
                                str(
                                    vec[
                                        "fill_color"
                                    ]
                                ),

                            "Width":
                                vec[
                                    "line_width"
                                ],

                            "BBox":
                                str(
                                    vec[
                                        "bbox"
                                    ]
                                )
                        })

                st.caption(
                    "Showing first 500 vector paths for performance."
                )

                st.dataframe(
                    pd.DataFrame(
                        vector_rows
                    ),
                    use_container_width=True
                )

            # ------------------------------------------------
            # IMAGES
            # ------------------------------------------------

            with tabs[5]:

                image_found = False

                for page in result[
                    "images"
                ]:

                    for image in page[
                        "data"
                    ]:

                        if "filename" not in image:
                            continue

                        image_found = True

                        path = os.path.join(
                            output_dir,
                            "embedded_images",
                            image[
                                "filename"
                            ]
                        )

                        st.write(
                            image[
                                "filename"
                            ]
                        )

                        try:

                            st.image(
                                path,
                                use_container_width=False
                            )

                        except Exception:

                            st.write(
                                "Image extracted but preview is unavailable."
                            )

                if not image_found:

                    st.info(
                        "No embedded images found."
                    )

            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            with tabs[6]:

                st.json(
                    master[
                        "metadata"
                    ]
                )

                st.write(
                    "SHA-256:"
                )

                st.code(
                    master[
                        "sha256"
                    ]
                )

                for page in master[
                    "pages"
                ]:

                    st.write(
                        f"Page "
                        f"{page['metadata']['page_number']}"
                    )

                    st.json(
                        page[
                            "metadata"
                        ]
                    )

            # ------------------------------------------------
            # RAW
            # ------------------------------------------------

            with tabs[7]:

                st.write(
                    "The complete raw data is available in the ZIP download below."
                )

                st.json({
                    "page_count":
                        master[
                            "page_count"
                        ],

                    "source":
                        master[
                            "source_file"
                        ],

                    "sha256":
                        master[
                            "sha256"
                        ]
                })

            # ------------------------------------------------
            # ZIP DOWNLOAD
            # ------------------------------------------------

            zip_path = os.path.join(
                temp_dir,
                "engineering_extraction.zip"
            )

            create_zip(
                output_dir,
                zip_path
            )

            with open(
                zip_path,
                "rb"
            ) as f:

                st.download_button(
                    "Download Full Extraction ZIP",
                    data=f,
                    file_name=(
                        "engineering_extraction.zip"
                    ),
                    mime=(
                        "application/zip"
                    )
                )

    # ========================================================
    # IMAGE
    # ========================================================

    else:

        st.subheader(
            "Image Drawing"
        )

        st.image(
            input_path,
            caption=(
                uploaded_file.name
            ),
            use_container_width=True
        )

        if st.button(
            "Extract Image Geometry"
        ):

            output_dir = os.path.join(
                temp_dir,
                "image_extracted"
            )

            with st.spinner(
                "Analyzing image..."
            ):

                result = (
                    extract_image_file(
                        input_path,
                        output_dir
                    )
                )

            st.success(
                "Image analysis completed."
            )

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                st.metric(
                    "Width",
                    result[
                        "width"
                    ]
                )

                st.metric(
                    "Height",
                    result[
                        "height"
                    ]
                )

            with col2:

                st.metric(
                    "Detected Lines",
                    result[
                        "detected_line_count"
                    ]
                )

            st.subheader(
                "Threshold View"
            )

            st.image(
                result[
                    "threshold_file"
                ],
                use_container_width=True
            )

            st.subheader(
                "Detected Edges"
            )

            st.image(
                result[
                    "edges_file"
                ],
                use_container_width=True
            )

            st.subheader(
                "Detected Lines"
            )

            st.image(
                result[
                    "line_preview_file"
                ],
                use_container_width=True
            )

            zip_path = os.path.join(
                temp_dir,
                "image_extraction.zip"
            )

            create_zip(
                output_dir,
                zip_path
            )

            with open(
                zip_path,
                "rb"
            ) as f:

                st.download_button(
                    "Download Image Extraction",
                    data=f,
                    file_name=(
                        "image_extraction.zip"
                    ),
                    mime=(
                        "application/zip"
                    )
                )


# ============================================================
# INFO
# ============================================================

st.divider()

st.caption(
    "Phase 1 extracts raw drawing information. "
    "Engineering symbol interpretation, route connectivity, "
    "OCR fallback and compliance checking should be added as separate stages."
)
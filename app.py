import streamlit as st
import os
import io
import zipfile
import base64
from pypdf import PdfReader, PdfWriter
from PIL import Image, ImageOps

def main():
    st.set_page_config(page_title="PDF Utility Tool", layout="wide")
    st.title("PDF Utility Tool")

    # Custom CSS for Tooltip
    st.markdown("""
    <style>
    .custom-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 5px;
    }
    .custom-tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #fff;
        color: #000;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        top: -5px;
        left: 110%; /* Position to the right */
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        font-weight: normal;
        border: 1px solid #ccc;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.2);
    }
    .custom-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["PDF分割 (Split)", "PDF結合 (Merge)", "画像をPDFに変換 (Image to PDF)", "PDF抽出 (Extract)"])

    with tab1:
        render_split_tab()
    
    with tab2:
        render_merge_tab()
        
    with tab3:
        render_image_to_pdf_tab()
        
    with tab4:
        render_extract_tab()

def render_split_tab():
    st.header("PDF分割 (Split)")
    uploaded_file = st.file_uploader("PDFファイルをアップロード", type="pdf", key="split_uploader")
    
    if uploaded_file:
        # PDF Preview logic
        try:
            reader = PdfReader(uploaded_file)
            total_pages = len(reader.pages)
            
            with st.expander("PDFプレビュー (ページ確認)", expanded=True):
                 def show_page(idx):
                     display_pdf_page(uploaded_file, idx)
                 
                 # Use filename as unique key prefix to reset state when file changes (simple hash-like)
                 # Actually, file_uploader key is constant, usage is safe.
                 render_preview_controls(total_pages, "split_preview", show_page)
        except Exception as e:
            st.error(f"プレビュー中にエラーが発生しました: {e}")

        mode = st.radio(
            "分割モードを選択:",
            ("全ページを1ページずつ分割", "ページ範囲を指定して抽出（例: 1-3, 5）")
        )
        
        range_input = ""
        if mode == "ページ範囲を指定して抽出（例: 1-3, 5）":
            range_input = st.text_input("ページ範囲 (例: 1-3, 5):", "1")

        if mode == "ページ範囲を指定して抽出（例: 1-3, 5）":
            range_input = st.text_input("ページ範囲 (例: 1-3, 5):", "1")

        compress = ui_compress_checkbox("split_compress")

        if st.button("分割実行", key="split_btn"):
            try:
                zip_buffer = split_pdf(uploaded_file, mode, range_input, compress)
                st.success("分割が完了しました！")
                st.download_button(
                    label="ZIPファイルをダウンロード",
                    data=zip_buffer,
                    file_name="split_pdfs.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

def render_preview_controls(total_items, key_prefix, display_callback):
    """
    Reusable component for pagination/preview.
    total_items: Total number of pages/items.
    key_prefix: Unique key prefix for session state (e.g., 'split_pdf').
    display_callback: Function(index) -> None, to render the content.
    """
    if total_items > 0:
        # State keys
        idx_key = f"{key_prefix}_current_index"
        input_key = f"{key_prefix}_input"
        
        # Initialize
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
            
        # Bounds check
        if st.session_state[idx_key] >= total_items:
            st.session_state[idx_key] = 0
            
        # Sync Input -> State
        if input_key in st.session_state:
             st.session_state[input_key] = st.session_state[idx_key] + 1
             
        col1, col2, col3, col4 = st.columns([1, 1.2, 0.5, 1])
        
        def on_change_input():
            st.session_state[idx_key] = st.session_state[input_key] - 1

        with col1:
            if st.button("⬅ 前へ", key=f"{key_prefix}_prev"):
                if st.session_state[idx_key] > 0:
                    st.session_state[idx_key] -= 1
                    st.session_state[input_key] = st.session_state[idx_key] + 1
                    st.rerun()

        with col2:
            st.number_input(
                "ページ",
                min_value=1,
                max_value=total_items,
                value=st.session_state[idx_key] + 1,
                key=input_key,
                on_change=on_change_input,
                label_visibility="collapsed"
            )
        
        with col3:
             st.markdown(f"<div style='margin-top: 10px; font-size: 18px;'>/ {total_items}</div>", unsafe_allow_html=True)

        with col4:
            if st.button("次へ ➡", key=f"{key_prefix}_next"):
                if st.session_state[idx_key] < total_items - 1:
                    st.session_state[idx_key] += 1
                    st.session_state[input_key] = st.session_state[idx_key] + 1
                    st.rerun()
                    
        # Render Content
        display_callback(st.session_state[idx_key])
    else:
        st.warning("プレビューできるアイテムがありません。")

def display_pdf_page(file, page_index):
    try:
        reader = PdfReader(file)
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        
        temp_buffer = io.BytesIO()
        writer.write(temp_buffer)
        temp_buffer.seek(0)
        
        base64_pdf = base64.b64encode(temp_buffer.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=0" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"ページ読み込みエラー: {e}")

def split_pdf(uploaded_file, mode, range_input, compress=False):
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)
    original_filename = os.path.splitext(uploaded_file.name)[0]
    
    # chunks will be a list of lists. Each inner list contains page indices (0-indexed) for one output PDF.
    chunks = []
    
    if mode == "全ページを1ページずつ分割":
        # Each page is a separate chunk
        for i in range(total_pages):
            chunks.append([i])
    else:
        # Parse range input "1-3, 5" -> chunks [[0, 1, 2], [4]]
        parts = [p.strip() for p in range_input.split(',')]
        for part in parts:
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    # User is 1-indexed, convert to 0-indexed
                    start_idx = max(0, start - 1)
                    end_idx = min(total_pages, end)
                    
                    if start_idx < end_idx:
                        # Add the range as a single chunk
                        chunks.append(list(range(start_idx, end_idx)))
                except ValueError:
                    pass # Ignore invalid format
            else:
                try:
                    idx = int(part) - 1
                    if 0 <= idx < total_pages:
                        chunks.append([idx])
                except ValueError:
                    pass

    if not chunks:
        raise ValueError("有効なページが選択されていません。")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, chunk_pages in enumerate(chunks):
            writer = PdfWriter()
            
            # Determine filename suffix
            # Logic: if chunk is single page -> _page_X
            # if chunk is range -> _pages_X-Y
            # However, since input ordering matters, user might want to know which split specific.
            # Simple naming:
            # If from "Split all" -> page_X
            # If from "Range" -> simply index? Or reconstruct name from chunk content?
            
            # Let's try to name it descriptively based on content
            start_p = chunk_pages[0] + 1
            end_p = chunk_pages[-1] + 1
            
            if len(chunk_pages) == 1:
                filename = f"{original_filename}_page_{start_p}.pdf"
            else:
                filename = f"{original_filename}_pages_{start_p}-{end_p}.pdf"

            for p_idx in chunk_pages:
                page = reader.pages[p_idx]
                if compress:
                    page.compress_content_streams()
                writer.add_page(page)
            
            pdf_bytes = io.BytesIO()
            writer.write(pdf_bytes)
            pdf_bytes.seek(0)
            
            # Handle duplicate filenames if user inputs "1-3, 1-3" (though unlikely/weird usage)
            # ZipFile allows duplicates but it's messy. We assume user is reasonable or we accept overwrites inside zip?
            # Better to ensure unique name in zip if collision?
            # For simplicity, append index if needed? 
            # Actually, standard zipfile.writestr will just add another entry.
            zf.writestr(filename, pdf_bytes.getvalue())

    zip_buffer.seek(0)
    return zip_buffer

def render_extract_tab():
    st.header("PDF抽出 (Extract)")
    st.markdown("指定したページのみを抽出し、**1つのPDFファイル**として保存します。")
    
    uploaded_file = st.file_uploader("PDFファイルをアップロード", type="pdf", key="extract_uploader")
    
    if uploaded_file:
         # PDF Preview logic (Reuse generic preview)
        try:
            reader = PdfReader(uploaded_file)
            total_pages = len(reader.pages)
            
            with st.expander("PDFプレビュー (ページ確認)", expanded=True):
                 def show_page(idx):
                     display_pdf_page(uploaded_file, idx)
                 render_preview_controls(total_pages, "extract_preview", show_page)
        except Exception as e:
            st.error(f"プレビュー中にエラーが発生しました: {e}")

        pages_input = st.text_input("抽出するページ番号 (例: 1, 3, 5-7)", "1")
        pages_input = st.text_input("抽出するページ番号 (例: 1, 3, 5-7)", "1")
        compress = ui_compress_checkbox("extract_compress")
        
        if st.button("抽出実行", key="extract_btn"):
            try:
                pdf_bytes = extract_pdf_pages(uploaded_file, pages_input, compress)
                st.success("抽出が完了しました！")
                st.download_button(
                    label="PDFファイルをダウンロード",
                    data=pdf_bytes,
                    file_name="extracted_pages.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

def extract_pdf_pages(uploaded_file, pages_input, compress=False):
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)
    
    pages_to_extract = []
    
    # Parse input "1, 3, 5-7"
    parts = [p.strip() for p in pages_input.split(',')]
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                start_idx = max(0, start - 1)
                end_idx = min(total_pages, end)
                if start_idx < end_idx:
                    pages_to_extract.extend(range(start_idx, end_idx))
            except ValueError:
                pass
        else:
            try:
                idx = int(part) - 1
                if 0 <= idx < total_pages:
                    pages_to_extract.append(idx)
            except ValueError:
                pass
    
    # Remove duplicates but KEEP ORDER? Req implies "construct 2 page PDF from p1 and p3".
    # Usually extract implies logical order of input. 
    # If user says "3, 1", they might expect page 3 then page 1.
    # I will allow duplicates and respect input order?
    # Req: "10ページのpdfがあったとして1ページ目と3ページ目だけで構成された全2ページのpdf"
    # -> Implies subsets.
    # I will unique them just in case, unless order matters. 
    # Let's keep order and allow duplicates? (Re-arranging pages is a valid use case).
    # But usually "Extraction" might imply just filtering. 
    # Let's simple unique and sort for safety, OR trust user?
    # "Extract" usually means "Pick these out". 
    # Let's trust user input order but remove invalid indices. 
    # (If user wants p1 then p3, pages_to_extract = [0, 2])
    
    if not pages_to_extract:
         raise ValueError("有効なページが選択されていません。")

    writer = PdfWriter()
    for p_idx in pages_to_extract:
        page = reader.pages[p_idx]
        if compress:
            page.compress_content_streams()
        writer.add_page(page)
        
    output_buffer = io.BytesIO()
    writer.write(output_buffer)
    output_buffer.seek(0)
    return output_buffer

def render_merge_tab():
    st.header("PDF結合 (Merge)")
    uploaded_files = st.file_uploader(
        "結合するPDFファイルを選択（複数可）", 
        type="pdf", 
        accept_multiple_files=True,
        key="merge_uploader"
    )

    if uploaded_files:
        with st.expander("PDFプレビュー (結合ファイルの確認)", expanded=True):
            # Select file to preview
            file_names = [f.name for f in uploaded_files]
            selected_file_name = st.selectbox("プレビューするファイルを選択:", file_names)
            
            # Find selected file object
            selected_file = next((f for f in uploaded_files if f.name == selected_file_name), None)
            
            if selected_file:
                try:
                    reader = PdfReader(selected_file)
                    total_pages = len(reader.pages)
                    
                    def show_page_merge(idx):
                        display_pdf_page(selected_file, idx)
                        
                    # Using dynamic key to reset state when file selection changes
                    safe_key = "".join(c for c in selected_file_name if c.isalnum())
                    render_preview_controls(total_pages, f"merge_{safe_key}", show_page_merge)
                except Exception as e:
                    st.error(f"プレビュー生成エラー: {e}")

    compress = ui_compress_checkbox("merge_compress")

    if uploaded_files and st.button("結合実行", key="merge_btn"):
        try:
            merged_pdf = merge_pdfs(uploaded_files, compress)
            st.success("結合が完了しました！")
            st.download_button(
                label="PDFファイルをダウンロード",
                data=merged_pdf,
                file_name="merged_output.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

def merge_pdfs(files, compress=False):
    writer = PdfWriter()
    
    for file in files:
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                if compress:
                    page.compress_content_streams()
                writer.add_page(page)
        except Exception as e:
            st.warning(f"ファイル {file.name} は破損しているか読み込めないためスキップされました: {e}")
            continue

    output_buffer = io.BytesIO()
    writer.write(output_buffer)
    output_buffer.seek(0)
    return output_buffer

def render_image_to_pdf_tab():
    st.header("画像をPDFに変換 (Image to PDF)")
    uploaded_images = st.file_uploader(
        "画像ファイルを選択 (JPG, PNG)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="img_uploader"
    )

    if uploaded_images:
        with st.expander("変換プレビュー (A4配置イメージ)", expanded=True):
            def show_img_preview(idx):
                img_file = uploaded_images[idx]
                st.caption(f"ファイル名: {img_file.name}")
                
                # Show processed A4 preview
                try:
                    img = Image.open(img_file)
                    processed_img = process_image_for_result_page(img)
                    st.image(processed_img, caption="A4変換プレビュー", use_column_width=True)
                except Exception as e:
                    st.error(f"画像処理エラー: {e}")

            render_preview_controls(len(uploaded_images), "img_preview", show_img_preview)

    if uploaded_images:
        compress = ui_compress_checkbox("img_compress")
        
    if uploaded_images and st.button("変換実行", key="img_btn"):
        try:
            pdf_output = convert_images_to_pdf(uploaded_images, compress)
            st.success("変換が完了しました！")
            st.download_button(
                label="PDFファイルをダウンロード",
                data=pdf_output,
                file_name="images_combined.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

def convert_images_to_pdf(images, compress=False):
    # A4 size in points (approx 72 dpi standard for PDF lib usage often)
    # 595 x 842 points
    A4_WIDTH, A4_HEIGHT = 595, 842
    
    writer = PdfWriter()
    
    for img_file in images:
        try:
            # Load image
            img = Image.open(img_file)
            
            # Handle Orientation
            img = ImageOps.exif_transpose(img)
            
            # Convert RGBA -> RGB (White bg)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3]) # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize logic: Fit to A4 maintaining aspect ratio
            img_w, img_h = img.size
            ratio_w = A4_WIDTH / img_w
            ratio_h = A4_HEIGHT / img_h
            scale = min(ratio_w, ratio_h)
            
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create white canvas
            canvas = Image.new('RGB', (A4_WIDTH, A4_HEIGHT), (255, 255, 255))
            
            # Center image
            x_offset = (A4_WIDTH - new_w) // 2
            y_offset = (A4_HEIGHT - new_h) // 2
            canvas.paste(img_resized, (x_offset, y_offset))
            
            # Save page to bytes
            page_bytes = io.BytesIO()
            canvas.save(page_bytes, format='PDF', optimize=compress)
            page_bytes.seek(0)
            
            # Append to PDF writer
            page_reader = PdfReader(page_bytes)
            page = page_reader.pages[0]
            if compress:
                page.compress_content_streams()
            writer.add_page(page)
            
        except Exception as e:
            st.warning(f"画像 {img_file.name} の処理中にエラーが発生したためスキップします: {e}")
            continue

    output_buffer = io.BytesIO()
    writer.write(output_buffer)
    output_buffer.seek(0)
    return output_buffer

def process_image_for_result_page(img):
    """
    Process image to A4 canvas (same logic as convert_images_to_pdf but returns PIL Image)
    """
    A4_WIDTH, A4_HEIGHT = 595, 842
    
    # Handle Orientation
    img = ImageOps.exif_transpose(img)
    
    # Convert RGBA -> RGB (White bg)
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize logic
    img_w, img_h = img.size
    ratio_w = A4_WIDTH / img_w
    ratio_h = A4_HEIGHT / img_h
    scale = min(ratio_w, ratio_h)
    
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white canvas
    canvas = Image.new('RGB', (A4_WIDTH, A4_HEIGHT), (255, 255, 255))
    
    # Center image
    x_offset = (A4_WIDTH - new_w) // 2
    y_offset = (A4_HEIGHT - new_h) // 2
    canvas.paste(img_resized, (x_offset, y_offset))
    
    return canvas

def ui_compress_checkbox(key):
    """
    Render a checkbox with a custom tooltip positioned to the right.
    """
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        checked = st.checkbox("データを圧縮して保存する", key=key)
    with col2:
        # Custom tooltip HTML
        st.markdown("""
        <div class="custom-tooltip" style="margin-top: 5px;">
            <div style="display:inline-block; border: 1px solid #ccc; border-radius: 50%; width: 20px; height: 20px; text-align: center; line-height: 18px; font-size: 14px; background-color: #fff; color: #000;">?</div>
            <span class="tooltiptext">ファイルサイズを小さくしたい場合はこちらを選んでください。</span>
        </div>
        """, unsafe_allow_html=True)
    return checked

if __name__ == "__main__":
    main()

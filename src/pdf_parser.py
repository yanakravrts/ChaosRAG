import fitz
import os
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class PDFParser:
    """
    A class to parse pdf files and extract text and images.
    """

    def __init__(self, pdf_path: str, output_dir: str = "data/processed"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir

    def _load_pdf(self) -> fitz.Document:
        """
        Load the pdf file into a fitz.Document object.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        if not self.pdf_path.endswith(".pdf"):
            raise ValueError(f"Invalid PDF file: {self.pdf_path}")
        return fitz.open(self.pdf_path)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the text by removing extra whitespace and newlines.
        """
        if not text:
            return ""
        return " ".join(text.split())

    def _extract_image(self, doc: fitz.Document, xref: int, page_num: int, img_index: int, min_image_size: int = 2048) -> Optional[str]:
        """
        Extract the image from pdf file.
        """
        base_image = doc.extract_image(xref)
        img = base_image["image"]

        if len(img) < min_image_size:
            return None

        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        img_path = os.path.join(images_dir, f"page_{page_num}_img_{img_index}.png")

        pix.save(img_path)

        return os.path.relpath(img_path, self.output_dir)

    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse the pdf file and extract text and images.
        """
        doc = self._load_pdf()
        data = []
        
        metadata = doc.metadata

        print(f"Processing: {self.pdf_path}")

        for page_num, page in enumerate(tqdm(doc, desc="Parsing pages")):
            real_page_num = page_num + 1
            
            raw_text = page.get_text()
            clean_text = self._clean_text(raw_text)
            
            image_paths = []
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                img_path = self._extract_image(doc, xref, real_page_num, img_index)
                if img_path:
                    image_paths.append(img_path)

            data.append({
                "page_number": real_page_num,
                "text": clean_text,
                "images": image_paths,
                "source": metadata.get('title', 'Unknown') or os.path.basename(self.pdf_path)
            })
            
        return data

    def save_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Save the data to a file.
        """
        os.makedirs(self.output_dir, exist_ok=True)            
        output_path = os.path.join(self.output_dir, "parsed_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":

    parser = PDFParser(pdf_path="data/raw/chaos_book.pdf", output_dir="data/processed")
    parsed_data = parser.parse()
    parser.save_data(parsed_data)

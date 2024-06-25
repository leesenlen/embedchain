import re
import os
import time
import copy
import hashlib
from typing import Optional, Any
import numpy as np
from PIL import Image
import pdfplumber
import requests
import logging
from embedchain.deepdoc.parser import PdfParser
from embedchain.rag.nlp import rag_tokenizer, naive_merge, tokenize_table, tokenize_chunks, tokenize, add_positions
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.helpers.json_serializable import register_deserializable


@register_deserializable
class PdfFileChunker(BaseChunker, PdfParser):
    """Chunker for PDF file."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        if config is None:
            config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=config.length_function,
        )
        super().__init__(text_splitter)

    def chunks(self, loader, src, metadata: Optional[dict[str, Any]] = None, config: Optional[ChunkerConfig] = None):
        self.pdf = pdfplumber.open(src) if isinstance(src, str) else pdfplumber.open(src)
        self.page_images = [p.to_image(resolution=72 * 3).annotated for i, p in enumerate(self.pdf.pages)]
        self.page_from = 0
        self.page_to = len(self.pdf.pages)
        documents = []
        chunk_ids = []
        idMap = {}
        min_chunk_size = config.min_chunk_size if config is not None else 1
        logging.info(f"[INFO] Skipping chunks smaller than {min_chunk_size} characters")

        if metadata is None:
            metadata = {}
        app_id = metadata.get("app_id", 1)
        knowledge_id = metadata.get("knowledge_id", 1)
        subject = metadata.get("subject", None)
        # OCR,布局识别
        sections, res, doc = self.ocr_and_layout_recognition(src)
        cks = self.chunk_with_layout(sections, res, config, doc)
        doc_id = self.generate_doc_id(app_id, "".join(each["content_with_weight"] for each in cks))
        metadatas = []
        for number, ck in enumerate(cks):
            # if ck.get("image"):
            #     # TODO 图片存储
            #     ...
            chunk = ck["content_with_weight"]
            chunk_id = str(doc_id) + "-" + hashlib.sha256(chunk.encode()).hexdigest()
            meta_data = {}
            url = src
            # add data type to meta data to allow query using data type
            meta_data["app_id"] = app_id
            meta_data["doc_id"] = doc_id
            meta_data["knowledge_id"] = knowledge_id
            meta_data["hash"] = doc_id
            meta_data["data_type"] = self.data_type.value
            # TODO: 主题可以是一段文本的summary
            meta_data["subject"] = subject if subject is not None else os.path.basename(url)
            meta_data["status"] = 1
            meta_data['segment_number'] = number
            if idMap.get(chunk_id) is None and len(chunk) >= min_chunk_size:
                idMap[chunk_id] = True
                chunk_ids.append(chunk_id)
                documents.append(f"主题：{meta_data['subject']}。段落内容：{chunk}")
                metadatas.append(meta_data)
        return {
            "documents": documents,
            "ids": chunk_ids,
            "metadatas": metadatas,
            "doc_id": doc_id
        }

    def ocr_and_layout_recognition(self, src: str):
        """
        调用sailvan_OCR进行行OCR解析，表格识别，布局识别。然后进行页面处理
        """
        ocr_url = os.getenv("OCR_URL")
        with open(src, "rb") as file:
            files = {
                "file_type": (None, "pdf"),  # (filename, filetype)
                "file": (src, file.read(), "application/pdf")
            }
        response = requests.post(ocr_url, files=files)
        assert response.status_code == 200, f"OCR failed: {response.text}"
        result = response.json()
        layout = result["boxes"]
        tbls = result["tables"]
        page_cum_height = result["page_cum_height"]
        doc = {
            "docnm_kwd": os.path.basename(src),
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", os.path.basename(src)))
        }
        sections = [(b["text"], self._line_tag(b, 3, page_cum_height)) for b in layout]
        res = tokenize_table(tbls, doc, False)
        return sections, res, doc

    @staticmethod
    def generate_doc_id(app_id, content):
        return str(app_id) + "-" + hashlib.sha256(content.encode()).hexdigest()

    def _line_tag(self, bx, ZM, page_cum_height):
        """
            分块文本添加页码，坐标等信息
        """
        pn = [bx["page_number"]]
        top = bx["top"] - page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return ""
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##" \
            .format("-".join([str(p) for p in pn]),
                    bx["x0"], bx["x1"], top, bott)

    def chunk_with_layout(self, sections, res, config, doc, delimiter="\n!?。；！？"):
        chunks = naive_merge(
            sections, config.chunk_size, delimiter)
        res.extend(self.tokenize_chunks(chunks, doc, False))
        return res

    def remove_tag(self, txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    def tokenize_chunks(self, chunks, doc, eng):
        res = []
        # wrap up as es documents
        for ck in chunks:
            if len(ck.strip()) == 0:
                continue
            print("--", ck)
            d = copy.deepcopy(doc)
            try:
                d["image"], poss = self.crop(ck, need_position=True)
                add_positions(d, poss)
                ck = self.remove_tag(ck)
            except NotImplementedError as e:
                pass
            tokenize(d, ck, eng)
            res.append(d)
        return res

    def crop(self, text, ZM=3, need_position=False):
        """
            将识别的布局快切割出来，用来保存缩略图，方便进行rag效果查看
        """
        imgs = []
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", text):
            pn, left, right, top, bottom = tag.strip(
                "#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(
                right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")],
                         left, right, top, bottom))
        if not poss:
            if need_position:
                return None, None
            return

        max_width = max(
            np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(
            0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + GAP),
                     min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + 120)))

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width
            bottom *= ZM
            for pn in pns[1:]:
                bottom += self.page_images[pn - 1].size[1]
            imgs.append(
                self.page_images[pns[0]].crop((left * ZM, top * ZM,
                                               right *
                                               ZM, min(
                    bottom, self.page_images[pns[0]].size[1])
                                               ))
            )
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(
                    bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                imgs.append(
                    self.page_images[pn].crop((left * ZM, 0,
                                               right * ZM,
                                               min(bottom,
                                                   self.page_images[pn].size[1])
                                               ))
                )
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, left, right, 0, min(
                        bottom, self.page_images[pn].size[1]) / ZM))
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB",
                        (width, height),
                        (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert('RGBA')
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic


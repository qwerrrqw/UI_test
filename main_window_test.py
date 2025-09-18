from __future__ import annotations
import os
import sys
import glob
import pathlib
import threading
import time
import json
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast
from pathlib import Path
from datetime import datetime, timedelta, timezone as _TZ   # ← timezone 추가(별칭은 _TZ)
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError        # ← 예외 추가
import socket
import psycopg2
import psycopg2.extras
import traceback
import shlex


from PySide6.QtCore import (
    Qt,
    QTimer,
    QThread,
    Signal,
    Slot,
    QDate,
    QRect
)
from PySide6.QtGui import QPixmap, QColor, QFontMetrics ,QBrush, QPalette ,QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QDateEdit,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QHeaderView,
    QAbstractItemView,
    QScrollArea,
    QStyledItemDelegate,
    QWidget,
    QFormLayout,
    QSizePolicy,
    QFrame,
    QStyle,
    QStyleOptionSpinBox,
    QCalendarWidget,
    QStyleOptionViewItem,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox
)
from PySide6.QtNetwork import QTcpSocket, QHostAddress
from datetime import date as _date

# --------------------------------------------------------------
# 외부 모듈: auto-generated UI + 암호화 설정 로더
# --------------------------------------------------------------
from ui_main_window import Ui_MainWindow  # noqa
from config_loader import load_encrypted_config  # noqa

# ---- Global QSS Loader ------------------------------------------------------
def apply_global_stylesheet(app: QApplication) -> None: ## CSS 전역 로더
    """
    전역 QSS를 여러 경로 후보에서 찾아 적용한다.
    우선순위:
    1) PyInstaller 임시폴더(_MEIPASS)/styles/style.css
    2) (frozen) 실행파일 옆 /styles/style.css
       (script) 이 파일(__file__) 폴더(ui)/styles/style.css
    3) (script) 이 파일(__file__) 폴더(ui)/style.css  ← ★ 너 지금 경로
    4) 프로젝트 루트(= ui 상위)/styles/style.css
    5) 현재 작업 디렉터리(CWD)/styles/style.css
    6) 환경변수 APP_STYLE_PATH(파일 또는 폴더)
    """
    def candidates():
        # 1) PyInstaller 임시폴더
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            yield Path(meipass) / "styles" / "style.css"

        # 2) 실행파일/스크립트 기준
        base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        yield base / "styles" / "style.css"   # ui/styles/style.css
        yield base / "style.css"              # ui/style.css  ← ★ 너의 현재 위치
        yield base.parent / "styles" / "style.css"  # 프로젝트루트/styles/style.css

        # 3) 현재 작업 디렉터리
        yield Path.cwd() / "styles" / "style.css"

        # 4) 환경변수
        env = os.getenv("APP_STYLE_PATH")
        if env:
            p = Path(env)
            yield p if p.suffix.lower() == ".css" else (p / "style.css")

    for p in candidates():
        try:
            if p.is_file():
                app.setStyleSheet(p.read_text(encoding="utf-8"))
                print(f"[style] loaded: {p}")
                return
        except Exception as e:
            print(f"[style] fail: {p} -> {e}")

    print("[style] no stylesheet found; running without custom QSS")

def strip_inline_styles(root: QWidget, keep: set[str] | None = None) -> None: ##
    """
    .ui에서 각 위젯에 박힌 인라인 setStyleSheet를 제거해서
    전역 style.css(QSS)가 적용되도록 만든다.
    keep: objectName을 넣으면 해당 위젯은 비우지 않음.
    """
    if keep is None:
        keep = set()

    # 자신
    try:
        if root.objectName() not in keep and root.styleSheet():
            root.setStyleSheet("")
    except Exception:
        pass

    # 자식들
    for w in root.findChildren(QWidget):
        try:
            if w.objectName() not in keep and w.styleSheet():
                w.setStyleSheet("")
        except Exception:
            continue

class AlignDelegate(QStyledItemDelegate): ## 중앙정렬 메서드
    def __init__(self, alignment: Qt.AlignmentFlag, parent=None):
        super().__init__(parent)
        self._alignment = alignment

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = self._alignment

def resolve_kst():
    """tzdata가 없어도 죽지 않게 Asia/Seoul 시도 후 KST(+09:00)로 폴백."""
    try:
        return ZoneInfo("Asia/Seoul")
    except ZoneInfoNotFoundError:
        return _TZ(timedelta(hours=9), name="KST")

def bg_color_for_status(text: str) -> QColor | None:
    """
    상태 텍스트에 맞는 배경색을 반환.
    - '작업 완료'/'OK' 계열 → 초록
    - 'NG'/'ERROR' 계열 → 빨강
    - '보류/HOLD/WARN' 계열 → 노랑
    매칭 없으면 None
    """
    t = (text or "").strip()
    if not t:
        return None
    if t in {"작업 완료", "완료", "정상"} or t.upper() in {"OK", "PASS"}:
        return QColor("#C8FEC8")  # green
    if t == "NG" or t.upper() in {"NG", "ERROR", "FAIL", "REJECT"}:
        return QColor("#FEC7C8")  # red
    if t in {"보류", "대기"} or t.upper() in {"HOLD", "WARN", "PENDING"}:
        return QColor("#FFC107")  # amber
    return None

# ---- 검색 헬퍼 ---------------------------------------------------
STATUS_SYNONYMS = {
    "ok": ["작업 완료"],
    "완료": ["작업 완료"],
    "ng": ["NG"],
    "fail": ["NG"],
    "에러": ["NG"],
}

def _escape_like(s: str) -> str:
    # LIKE/ILIKE 패턴용 이스케이프:
    # - \, %, _ 를 파라미터 문자열 내에서 안전하게 이스케이프
    # ※ SQL에 literal로 넣지 않고 항상 %s 바인딩으로 전달하므로 ESCAPE 절 불필요
    s = s.replace("\\", "\\\\")          # backslash 자체 이스케이프
    s = s.replace("%", r"\%").replace("_", r"\_")
    return s

def _tokenize(query: str) -> list[str]:
    """
    공백 AND, 따옴표로 phrase 유지:
    ex)  abc "foo bar" -bad pid:123 x|y
    """
    try:
        toks = shlex.split(query, posix=True)
    except Exception:
        toks = query.split()
    return [t for t in (tok.strip() for tok in toks) if t]

def _split_field_token(tok: str) -> tuple[str|None, str]:
    """
    'pid:123', 'destination:ok' 같은 필드 지정 문법 파싱.
    반환: (field_alias or None, value)
    """
    if ":" in tok:
        k, v = tok.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k:
            return k, v
    return None, tok

def _alts(val: str) -> list[str]:
    """토큰 내 OR 지원: 'a|b|c' → ['a','b','c']"""
    return [a for a in (p.strip() for p in val.split("|")) if a]

def _barcode_expr(col: str) -> str:
    """
    barcode 특수 처리: 하이픈/공백 제거 후 부분일치도 허용
    (정규식 치환 사용, ESCAPE 절 불필요)
    """
    return f"regexp_replace({col}, '[-\\s]', '', 'g') ILIKE %s"

def _build_keyword_filters(kw: str, search_cols: list[str], colnames: set[str]):
    """
    공백 분리 다중 토큰 + 따옴표 묶음 + 제외(-토큰) + OR(|) + 대소문자 무시(ILIKE)
    - field alias: pid:, sscc:, destination:/status:, barcode:, error_reason:
    - barcode는 하이픈/공백 제거 비교를 추가로 수행
    - status 동의어(OK/NG 등) 확장 지원
    반환: (where_parts: list[str], params: list[Any])
    """
    tokens = _tokenize(kw)
    where_parts: list[str] = []
    params: list[str] = []

    # UI에서 'Destination' 이 실제 status 컬럼을 의미
    alias_map = {
        "pid": "pid",
        "sscc": "sscc",
        "destination": "status",
        "status": "status",
        "barcode": "barcode",
        "error_reason": "error_reason",
    }

    def build_like_set(cols: list[str], one_val: str, negate: bool) -> tuple[str, list[str]]:
        """
        한 개 값(one_val)을 여러 컬럼(cols)에 적용:
        - 포지티브: (col1 LIKE ? OR col2 LIKE ? ...)
        - 네거티브: NOT(...) 을 alt 단위로 만들고 AND로 묶음
        - barcode는 normalize 비교 + 일반 ILIKE 둘다 추가
        """
        safe = _escape_like(one_val)
        pat = f"%{safe}%"

        # 이 값에 대한 '한 번의' 컬럼 집합식 (OR)
        col_exprs: list[str] = []
        col_params: list[str] = []

        for c in cols:
            if c == "barcode" and "barcode" in colnames and one_val:
                norm = one_val.replace("-", "").replace(" ", "")
                col_exprs.append(_barcode_expr("barcode"))
                col_params.append(f"%{_escape_like(norm)}%")
                col_exprs.append(f"{c} ILIKE %s")
                col_params.append(pat)
            else:
                col_exprs.append(f"{c} ILIKE %s")
                col_params.append(pat)

        expr = "(" + " OR ".join(col_exprs) + ")"
        if negate:
            # 이 값(one_val)이 어떤 컬럼에도 나타나면 안 된다 → NOT (col1 OR col2 ...)
            return f"NOT {expr}", col_params
        else:
            return expr, col_params

    for raw in tokens:
        negate = raw.startswith("-")
        tok = raw[1:].strip() if negate else raw
        if not tok:
            continue

        field_alias, val = _split_field_token(tok)
        alt_vals = _alts(val)
        if not alt_vals:
            continue

        # 이번 토큰에 적용할 컬럼
        if field_alias:
            mapped = alias_map.get(field_alias)
            cols = [mapped] if mapped else list(search_cols)
        else:
            cols = list(search_cols)

        # status 동의어 확장
        if any(c == "status" for c in cols):
            expanded: list[str] = []
            for a in alt_vals:
                k = a.lower()
                expanded.extend(STATUS_SYNONYMS.get(k, [a]))
            alt_vals = expanded

        # alt (a|b|c) 를 묶는 규칙:
        #  - 포지티브: ( alt(a) OR alt(b) OR alt(c) )
        #  - 네거티브: ( NOT alt(a) AND NOT alt(b) AND NOT alt(c) )
        alt_group_exprs: list[str] = []
        alt_group_params: list[str] = []
        for a in alt_vals:
            expr, p = build_like_set(cols, a, negate)
            alt_group_exprs.append(expr)
            alt_group_params.extend(p)

        if negate:
            where_parts.append("(" + " AND ".join(alt_group_exprs) + ")")
        else:
            where_parts.append("(" + " OR ".join(alt_group_exprs) + ")")
        params.extend(alt_group_params)

    return where_parts, params

# ---- 로고 경로 ----
def resource_path(*parts: str) -> str:
    """여러 베이스 경로 후보를 순회하며 첫 번째로 존재하는 파일 경로를 반환."""
    rel = Path(*parts)

    candidates = []

    # 1) PyInstaller 런타임 임시 폴더
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / rel)

    # 2) 이 파일이 있는 폴더 / 그 부모들
    here = Path(__file__).resolve().parent
    candidates += [
        here / rel,
        here.parent / rel,
        here.parent.parent / rel,
    ]

    # 3) 실행 파일 있는 폴더, 현재 작업 폴더
    candidates += [
        Path(sys.argv[0]).resolve().parent / rel,
        Path.cwd() / rel,
    ]

    # 존재 확인
    for c in candidates:
        if c.exists():
            return str(c)

    # 못 찾으면 디버그용으로 후보들을 찍고, 마지막 후보를 돌려준다(문구 표시용)
    try:
        print("[resource_path] not found. tried:")
        for c in candidates:
            print("  -", c)
    except Exception:
        pass
    return str(candidates[-1] if candidates else rel)

# ---- 호환 상수: StateFlag 없으면 State로 폴백 ----
StateEnum = getattr(QStyle, "StateFlag", QStyle)
STATE_MOUSEOVER = getattr(StateEnum, "State_MouseOver")
STATE_SELECTED  = getattr(StateEnum, "State_Selected")

class StatusBGDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        # 처리 상태 컬럼 정렬을 가운데로
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter

    def paint(self, painter, option, index):
        text = (index.data() or "").strip()
        bg = bg_color_for_status(text)

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        opt.state = opt.state & ~STATE_MOUSEOVER

        if bg and not (opt.state & STATE_SELECTED):
            o = cast(Any, opt)                 # ✅ 타입 체커 무시
            painter.save()
            painter.fillRect(o.rect, bg)       # rect 경고 사라짐
            painter.restore()
            o.palette.setColor(QPalette.ColorRole.Text, QColor("#00000"))  # palette 경고 사라짐

        super().paint(painter, opt, index)

def apply_status_cell_colors(item: QTableWidgetItem, *, text_color: str = "#000000") -> None:
    """
    상태 문자열(item.text())를 보고 배경색을 칠하고,
    글자색은 기본 검정(또는 지정)으로 통일한다.
    bg_color_for_status(text) → QColor | None 이라고 가정.
    """
    text = (item.text() or "").strip()
    bg = bg_color_for_status(text)
    if bg is not None:
        item.setBackground(QBrush(bg))
        item.setForeground(QBrush(QColor(text_color)))
    else:
        # 필요하면 기본으로 되돌리기 (선택)
        item.setBackground(QBrush())   # clear
        item.setForeground(QBrush())   # clear

# ---------------------------------------------------------------------------
# 데이터 모델 정의
# ---------------------------------------------------------------------------
@dataclass
class DBConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    table_jobs: str  # 최근 20건, 전체 조회용 테이블명

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DBConfig":
        """
        암호화 설정에서 DB 접속정보를 추출.
        - 신형(nested) 포맷: cfg['db']['host' | 'port' | ...]
        - 구형(flat) 포맷:  cfg['DB_HOST' | 'DB_PORT' | 'DB_NAME' | 'DB_USER' | 'DB_PASSWORD']
        """
        db_nested = cfg.get("db", {})
        tbl_nested = cfg.get("table", {})

        # --- flat 키 추출 (구형 포맷 호환) ---
        flat_host = cfg.get("DB_HOST")
        flat_port = cfg.get("DB_PORT")
        flat_name = cfg.get("DB_NAME")
        flat_user = cfg.get("DB_USER")
        flat_pass = cfg.get("DB_PASSWORD")
        flat_table = cfg.get("DB_TABLE") or cfg.get("JOBS_TABLE")

        # host
        host = db_nested.get("host") or flat_host or "127.0.0.1"

        # port (정수 변환)
        port = db_nested.get("port") or flat_port or 5432
        try:
            port = int(port)
        except Exception:
            port = 5432

        # dbname
        dbname = (
                db_nested.get("dbname")
                or db_nested.get("name")
                or flat_name
                or "postgres"
        )

        # user
        user = db_nested.get("user") or flat_user or "postgres"

        # password
        password = (
                db_nested.get("password")
                or db_nested.get("passwd")
                or flat_pass
                or ""
        )

        # jobs 테이블명
        table_jobs = tbl_nested.get("jobs") or tbl_nested.get("table") or flat_table or "tb_send"

        return cls(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            table_jobs=table_jobs,
        )


def qdate_to_date(qd) -> _date:
    """PySide QDate → Python date 안전 변환."""
    # PySide6 >=6.7? qd.toPython() 존재하지만 호환성 위해 수동 변환
    try:
        return qd.toPython()  # 일부 버전에서 정상 동작
    except Exception:
        pass
    return _date(qd.year(), qd.month(), qd.day())


@dataclass
class PathConfig:
    mapped_image_root: pathlib.Path

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PathConfig":
        paths = cfg.get("paths", {})
        root = paths.get("mapped_image_root", r"D:/mapped_image")
        return cls(mapped_image_root=pathlib.Path(root))



@dataclass
class AppConfig:
    center_name: str
    db: DBConfig
    paths: PathConfig
    poll_interval_sec: float = 1.0  # DB 주기
    image_poll_interval_sec: float = 3.0  # 이미지 주기

    @classmethod
    def load(cls, enc_path: str = "path.enc") -> "AppConfig":
        cfg = load_encrypted_config(enc_path=enc_path)
        return cls(
            center_name=cfg.get("center_name", "센터 미지정"),
            db=DBConfig.from_config(cfg),
            paths=PathConfig.from_config(cfg),
            poll_interval_sec=float(cfg.get("poll_interval_sec", 1.0)),
            image_poll_interval_sec=float(cfg.get("image_poll_interval_sec", 3.0)),
        )


# ---------------------------------------------------------------------------
# DB Poller Thread
# ---------------------------------------------------------------------------
class DBPollerThread(QThread):
    """주기적으로 DB에서 오늘 건수 및 최신 20건을 읽어오는 쓰레드."""

    daily_count_signal = Signal(int)
    latest_rows_signal = Signal(list)  # list[dict]
    db_error_signal = Signal(str)

    def __init__(self, cfg: AppConfig, tz: ZoneInfo, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.tz = tz
        self._stop = threading.Event()
        self._conn = None  # type: Optional[psycopg2.extensions.connection]

    # psycopg2 연결
    def _connect(self):
        try:
            self._conn = psycopg2.connect(
                host=self.cfg.db.host,
                port=self.cfg.db.port,
                dbname=self.cfg.db.dbname,
                user=self.cfg.db.user,
                password=self.cfg.db.password,
                connect_timeout=5,
            )
            self._conn.autocommit = True
        except Exception as e:  # pragma: no cover - 연결 실패 로깅용
            self.db_error_signal.emit(f"DB 연결 실패: {e}")
            self._conn = None

    def stop(self):
        self._stop.set()

    # 오늘 자정~내일 자정 범위
    def _today_range(self) -> Tuple[datetime, datetime]:
        now = datetime.now(self.tz)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end

    def _fetch_daily_count(self, cur) -> Optional[int]:
        table = self.cfg.db.table_jobs
        start, end = self._today_range()
        # created_at 컬럼 이름은 프로젝트별 상이 → 후보 리스트 중 존재하는 컬럼 찾기
        # 가장 흔한 이름: created_at, ts, timestamp, created, dt
        column_candidates = ["created_at", "ts", "timestamp", "created", "dt"]
        col = None
        # 스키마 검사 한 번만 해도 되지만 간단히 try 순차 실행
        for c in column_candidates:
            try:
                cur.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {c} >= %s AND {c} < %s",
                    (start, end),
                )
                col = c
                break
            except Exception:  # ignore, 컬럼 없음
                self._conn.rollback()
        if col is None:
            return None
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _fetch_latest_rows(self, cur, limit: int = 20) -> List[Dict[str, Any]]:
        table = self.cfg.db.table_jobs
        query_order = [
            # id 있는 경우: 시간 필드 추가(created_at)
            ("id",
             f"SELECT id, pid, sscc, barcode, status AS destination, error_reason, created_at FROM {table} ORDER BY id DESC LIMIT %s"),
            # created_at 대체 경우
            ("created_at",
             f"SELECT pid, sscc, barcode, status AS destination, error_reason, created_at FROM {table} ORDER BY created_at DESC LIMIT %s"),
        ]

        for col, q in query_order:
            try:
                cur.execute(q, (limit,))
                cols = [d.name for d in cur.description]
                rows = cur.fetchall()
                return [dict(zip(cols, r)) for r in rows]
            except Exception:
                # 해당 컬럼(또는 쿼리) 실패 시 롤백 후 다음 후보 시도
                self._conn.rollback()

        return []

    def run(self):  # noqa: D401
        self._connect()
        if self._conn is None:
            return
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        poll = self.cfg.poll_interval_sec
        while not self._stop.is_set():
            try:
                count = self._fetch_daily_count(cur)
                if count is not None:
                    self.daily_count_signal.emit(count)
                rows = self._fetch_latest_rows(cur, 20)
                self.latest_rows_signal.emit(rows)
            except Exception as e:
                self.db_error_signal.emit(str(e))
            finally:
                time.sleep(poll)
        cur.close()
        if self._conn:
            self._conn.close()


# ---------------------------------------------------------------------------
# 전체 데이터 팝업 다이얼로그
# ---------------------------------------------------------------------------
class AllDataDialog(QDialog):
    """전체 DB 레코드 보기 + 필터."""

    class _NumericItem(QTableWidgetItem):
        def __lt__(self, other):
            try:
                a = int(self.text().replace(',', ''))
                b = int(other.text().replace(',', ''))
                return a < b
            except Exception:
                return super().__lt__(other)

    class _NoStepDateEdit(QDateEdit):
        """마우스 클릭/휠로 값이 +1 되는 것을 막고, 캘린더 버튼만 동작하게."""

        def wheelEvent(self, e):  # 휠로 증감 방지
            e.ignore()

        def stepBy(self, steps):  # 키/버튼 스텝 자체 차단(캘린더 선택은 영향 없음)
            return

        def mousePressEvent(self, e):
            # 캘린더/스핀 버튼 영역 클릭만 통과, 나머지 빈 영역 클릭은 증감 차단
            opt = QStyleOptionSpinBox()
            opt.initFrom(self)
            opt.buttonSymbols = self.buttonSymbols()
            up_rect = self.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, opt, QStyle.SubControl.SC_SpinBoxUp, self)
            down_rect = self.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, opt, QStyle.SubControl.SC_SpinBoxDown, self)
            if up_rect.contains(e.pos()) or down_rect.contains(e.pos()):
                return super().mousePressEvent(e)  # (캘린더 팝업 버튼 포함) 버튼 클릭은 허용
            # 빈 배경 클릭은 포커스만 주고 증감은 막기
            self.setFocus()
            e.accept()

    def __init__(self, cfg: AppConfig, tz: ZoneInfo, parent=None): ##
        super().__init__(parent)
        self.cfg = cfg
        self.tz = tz
        self.setWindowTitle("작업 전체 조회")

        ## 스타일 스코프용 objectName 부여(중요)
        self.setObjectName("AllDataDialog")

        # 위젯(이 아래 부분은 위젯의 '구조'와 '동작'에 관한 것이므로 분리 대상이 아님)
        layout = QVBoxLayout(self)

        # 필터 행
        filter_row = QHBoxLayout()
        # self.date_from = QDateEdit(self)
        self.date_from = self._NoStepDateEdit(self) # 변경
        self.date_from.setCalendarPopup(True)
        # self.date_to = QDateEdit(self)
        self.date_to = self._NoStepDateEdit(self) # 변경
        self.date_to.setCalendarPopup(True)
        today = QDate.currentDate()
        self.date_from.setDate(today)
        self.date_to.setDate(today)

        # __init__ 안쪽, date_from/date_to 만든 직후에 추가
        for de in (self.date_from, self.date_to):
            self._inflate_calendar_popup(de)

        self.search_field = QComboBox(self)
        self.search_field.addItems(["PID", "SSCC", "Destination", "barcode", "전체"])
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("검색어...")
        self.search_btn = QPushButton("검색", self)
        self.search_btn.clicked.connect(self._on_search_clicked)

        filter_row.addWidget(QLabel("시작일:"))
        filter_row.addWidget(self.date_from)
        filter_row.addWidget(QLabel("종료일:"))
        filter_row.addWidget(self.date_to)
        filter_row.addWidget(self.search_field)
        filter_row.addWidget(self.search_input)
        filter_row.addWidget(self.search_btn)
        layout.addLayout(filter_row)

        # 테이블
        self.table = QTableWidget(self)
        self.table.setColumnCount(8)  # 컬럼 8개로 증가
        self.table.setHorizontalHeaderLabels(
            ["No.", "PID", "SSCC", "Destination", "barcode", "error_reason", "이미지", "시간"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setItemDelegateForColumn(3, StatusBGDelegate(self.table))

        # 기본 헤더 라벨 저장
        self._base_headers = ["No.", "PID", "SSCC", "Destination", "barcode", "error_reason", "이미지", "시간"]

        # 정렬 관련 상태  ←★ 먼저 정의해 둡니다
        self._sorting_guard = False
        self._allowed_sort_cols = {0, 1, 7}  # No., PID, 시간
        self._sort_col = -1  # 현재 정렬 컬럼 없음
        self._sort_order = Qt.SortOrder.AscendingOrder

        # 헤더를 item 기반으로 교체(▲/▼ 텍스트 갱신하기 위해)
        for i, text in enumerate(self._base_headers):
            self.table.setHorizontalHeaderItem(i, QTableWidgetItem(text))

        # 줄무늬 등 각종 설정 (생략 가능)
        self.table.setAlternatingRowColors(True)

        # 헤더/정렬 연결
        hdr = self.table.horizontalHeader()
        hdr.setSortIndicatorShown(True)
        self.table.setSortingEnabled(False)  # 기본 자동정렬은 끔
        hdr.sectionClicked.connect(self._on_header_clicked)

        # ✅ 기본 정렬: 시간(컬럼 7) 내림차순
        self._sort_col = 7
        self._sort_order = Qt.SortOrder.DescendingOrder
        hdr.setSortIndicator(self._sort_col, self._sort_order)
        self._update_header_sort_icons()

        # ★ 이제 아이콘(▲/▼) 초기 반영
        self._update_header_sort_icons()

        # (이하 칼럼 폭, 버튼, DB 연결, _load_all() 호출 등 기존 그대로)

        # 더블클릭 이벤트 연결
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        # 테이블 컬럼 너비 설정 (이것은 스타일보다는 '동작 방식'에 가까움)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # No.
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)  # PID
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)  # SSCC
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)  # Destination
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # barcode
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # error_reason
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)  # 이미지
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Interactive)  # 시간

        # 초기 컬럼 너비 설정
        ## (여러 테이블에서 일관된 너비 정책을 쓸 거면 JSON 등 외부 설정으로 분리 고려. QSS로 직접 제어는 어려움)
        self.table.setColumnWidth(0, 50)  # No.
        self.table.setColumnWidth(1, 100)  # PID
        self.table.setColumnWidth(2, 100)  # SSCC
        self.table.setColumnWidth(3, 100)  # Destination
        self.table.setColumnWidth(6, 150)  # 이미지
        self.table.setColumnWidth(7, 150)  # 시간

        layout.addWidget(self.table)

        # CSV 내보내기 호출 버튼
        export_btn = QPushButton("CSV 내보내기", self)
        export_btn.clicked.connect(self._export_to_csv)

        # 버튼이 가로로 늘어나지 않도록
        export_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        export_btn.setFixedWidth(120)  # 폭은 취향대로 (예: 100~140)

        # 바로 오른쪽 정렬로 추가
        layout.addWidget(export_btn, 0, Qt.AlignmentFlag.AlignRight)
        # 위젯들 다 만든 뒤에 호출 (가장 마지막쯤)
        self._resize_to_ratio(w_ratio=0.70, h_ratio=0.50)

        # DB 연결(열 때 바로 한번)
        self._conn = None
        self._connect_db()
        self._load_all()

    def _on_header_clicked(self, col: int):
        if self._sorting_guard:
            return
        self._sorting_guard = True
        try:
            hdr = self.table.horizontalHeader()

            # 허용 안 된 컬럼이면 기존 표시만 유지
            if col not in self._allowed_sort_cols:
                if self._sort_col >= 0:
                    hdr.setSortIndicator(self._sort_col, self._sort_order)
                else:
                    hdr.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
                self._update_header_sort_icons()
                return

            same = (self._sort_col == col)

            # 정렬 순서 결정: 같은 컬럼 재클릭이면 토글, 다른 컬럼이면 ASC 시작
            self._sort_order = (
                Qt.SortOrder.DescendingOrder
                if same and self._sort_order == Qt.SortOrder.AscendingOrder
                else Qt.SortOrder.AscendingOrder
            )

            # 컬럼 선택: 다른 컬럼 클릭 시에만 변경
            if not same:
                self._sort_col = col

            hdr.setSortIndicator(self._sort_col, self._sort_order)
            self.table.setSortingEnabled(True)
            self.table.sortItems(self._sort_col, self._sort_order)
            self.table.setSortingEnabled(False)
            self._update_header_sort_icons()
        finally:
            self._sorting_guard = False

    def _resize_to_ratio(self, w_ratio: float = 0.70, h_ratio: float = 0.70) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if not screen:
            return
        avail = screen.availableGeometry()
        w = int(avail.width() * w_ratio)
        h = int(avail.height() * h_ratio)

        self.resize(w, h)

        # 중앙 정렬
        g = self.frameGeometry()
        g.moveCenter(avail.center())
        self.move(g.topLeft())

    # 클래스 메서드로 추가
    def _inflate_calendar_popup(self, de: QDateEdit):
        cal = QCalendarWidget(self)
        cal.setObjectName("AllDataCalendar")  # QSS 타깃용
        cal.setGridVisible(True)  # 날짜 격자 보이기
        cal.setMinimumSize(380, 320)  # 팝업 자체를 조금 크게
        # 폰트 키우기
        f = cal.font()
        f.setPointSize(max(14, f.pointSize() + 2))
        cal.setFont(f)
        # 이 커스텀 달력을 팝업으로 사용
        de.setCalendarWidget(cal)

        # 입력 필드 자체 클릭영역도 살짝 키우기(선택)
        # de.setMinimumHeight(34)

    def _update_header_sort_icons(self):
        """현재 self._sort_col / self._sort_order 상태를 헤더 텍스트(▲/▼)에 반영."""
        for i, base in enumerate(self._base_headers):
            text = base
            if i == self._sort_col and i in self._allowed_sort_cols:
                arrow = "▲" if self._sort_order == Qt.SortOrder.AscendingOrder else "▼"
                text = f"{base} {arrow}"
            item = self.table.horizontalHeaderItem(i)
            if item is None:
                item = QTableWidgetItem(text)
                self.table.setHorizontalHeaderItem(i, item)
            else:
                item.setText(text)

    def _connect_db(self):
        try:
            self._conn = psycopg2.connect(
                host=self.cfg.db.host,
                port=self.cfg.db.port,
                dbname=self.cfg.db.dbname,
                user=self.cfg.db.user,
                password=self.cfg.db.password,
                connect_timeout=5,
            )
            self._conn.autocommit = True
        except Exception as e:
            QMessageBox.critical(self, "DB 오류", f"DB 연결 실패: {e}")

    def _exec_query(self, query: str, params: Tuple[Any, ...] = ()):  # -> list[dict]
        if self._conn is None:
            return []
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query, params)
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:
            QMessageBox.critical(self, "DB 오류", str(e))
            return []

    def _on_search_clicked(self):
        self._load_all()

    def _on_cell_double_clicked(self, row, col):
        # 이미지 경로 컬럼(6번)을 더블 클릭했을 때만 처리
        if col == 6:
            # ✅ [NEW] 아이템/셀위젯 모두 대응해서 경로 얻기
            w = self.table.cellWidget(row, col)
            item = self.table.item(row, col)
            image_path = (w.text().strip() if isinstance(w, QLabel) else (item.text().strip() if item else ""))

            # ↓↓↓ 기존 로직은 그대로 유지 ↓↓↓
            if not image_path:
                QMessageBox.information(self, "알림", "이미지 경로가 없습니다.")
                return

            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "경고", f"이미지 파일을 찾을 수 없습니다:\n{image_path}")
                return

            # PyQt 이미지 뷰어 대화 상자로 열기
            try:
                # 이미지 로드
                original_pixmap = QPixmap(image_path)
                if original_pixmap.isNull():
                    QMessageBox.warning(self, "오류", "이미지를 로드할 수 없습니다.")
                    return

                # 이미지 뷰어 대화 상자 생성
                image_dialog = QDialog(self)
                image_dialog.setWindowTitle(f"이미지 뷰어 - {os.path.basename(image_path)}")
                image_dialog.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)  # 항상 위에 표시
                # ✅ QSS 스코프용 objectName
                image_dialog.setObjectName("ImageViewerDialog")
                # 레이아웃 설정
                layout = QVBoxLayout(image_dialog)

                # 스크롤 영역 생성
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)

                # 화면 크기 가져오기
                screen_size = QApplication.primaryScreen().size()
                max_width = int(screen_size.width() * 0.7)  # 화면 너비의 70%
                max_height = int(screen_size.height() * 0.7)  # 화면 높이의 70%

                # 이미지 크기 조정 (원본 비율 유지)
                pixmap = original_pixmap
                if original_pixmap.width() > max_width or original_pixmap.height() > max_height:
                    pixmap = original_pixmap.scaled(
                        max_width, max_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )

                # 확대/축소 상태 저장
                zoom_level = 20

                # 이미지 표시용 라벨
                image_label = QLabel()
                image_label.setPixmap(pixmap)

                ## ⛔ 인라인 정렬 → QSS로 이동 (필요 시 다시 켜도 됨)
                # image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                ## ✅ QSS 타깃팅을 위해 ID 지정
                image_label.setObjectName("ImageLabel")

                # 스크롤 영역에 라벨 추가
                scroll_area.setWidget(image_label)

                # 확대/축소 버튼을 위한 레이아웃
                button_layout = QHBoxLayout()

                # 확대 버튼
                zoom_in_btn = QPushButton("확대(+)")
                ## ⛔ 인라인 고정폭 → QSS(min/max-width)로 이동
                # zoom_in_btn.setFixedWidth(100)

                # 축소 버튼
                zoom_out_btn = QPushButton("축소(-)")
                ## zoom_out_btn.setFixedWidth(100)

                # 원본 크기 버튼
                original_btn = QPushButton("초기화")
                ## original_btn.setFixedWidth(100)

                # 현재 확대/축소 레벨 표시
                zoom_level_label = QLabel(f"확대: {zoom_level}%")
                # (선택) QSS에서 스타일 주려면 ID 부여
                zoom_level_label.setObjectName("ZoomLevelLabel")

                # 닫기 버튼
                close_btn = QPushButton("닫기")
                ## close_btn.setFixedWidth(100)

                # --- 이 아래는 버튼의 '기능'과 관련된 로직이므로 분리 대상이 아님 ---
                # 확대 기능
                def zoom_in():
                    nonlocal zoom_level, pixmap
                    if zoom_level < 300:  # 최대 300%까지 확대
                        zoom_level += 20
                        update_zoom()

                # 축소 기능
                def zoom_out():
                    nonlocal zoom_level, pixmap
                    if zoom_level > 20:  # 최소 20%까지 축소
                        zoom_level -= 20
                        update_zoom()

                # 원본 크기로 보기
                def show_original():
                    nonlocal zoom_level
                    zoom_level = 20
                    update_zoom()

                # 확대/축소 레벨 업데이트
                def update_zoom():
                    nonlocal zoom_level
                    zoom_factor = zoom_level / 100.0
                    new_pixmap = original_pixmap.scaled(
                        int(original_pixmap.width() * zoom_factor),
                        int(original_pixmap.height() * zoom_factor),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    image_label.setPixmap(new_pixmap)
                    zoom_level_label.setText(f"확대: {zoom_level}%")

                # 버튼 이벤트 연결
                zoom_in_btn.clicked.connect(zoom_in)
                zoom_out_btn.clicked.connect(zoom_out)
                original_btn.clicked.connect(show_original)
                close_btn.clicked.connect(image_dialog.accept)

                # 버튼 레이아웃에 위젯 추가
                button_layout.addWidget(zoom_in_btn)
                button_layout.addWidget(zoom_out_btn)
                button_layout.addWidget(original_btn)
                button_layout.addWidget(zoom_level_label)
                button_layout.addStretch()
                button_layout.addWidget(close_btn)

                # 메인 레이아웃에 위젯 추가
                layout.addWidget(scroll_area)
                layout.addLayout(button_layout)

                # 대화 상자 크기 설정
                image_dialog.resize(
                    min(pixmap.width() + 50, max_width + 50),
                    min(pixmap.height() + 100, max_height + 100)
                )

                # 대화 상자 표시
                image_dialog.exec()

            except Exception as e:
                QMessageBox.critical(self, "오류", f"이미지 열기 실패: {str(e)}")

    def _export_to_csv(self):
        """현재 테이블 데이터를 CSV 파일로 내보내기(선택한 컬럼만)."""
        # 1) 컬럼 선택
        cols = self._choose_export_columns()
        if cols is None:
            return  # 취소
        if not cols:
            QMessageBox.information(self, "안내", "선택된 컬럼이 없습니다.")
            return

        # 2) 파일 저장 대화상자
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "CSV 파일 저장",
            "",
            "CSV 파일 (*.csv);;모든 파일 (*.*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        # 3) 쓰기
        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f)

                # 헤더: 선택한 컬럼만 + 정렬 화살표 제거
                headers = []
                for c in cols:
                    it = self.table.horizontalHeaderItem(c)
                    text = it.text() if it else ""
                    text = text.replace(" ▲", "").replace(" ▼", "")
                    headers.append(text)
                wr.writerow(headers)

                # 데이터: 선택한 컬럼만
                for r in range(self.table.rowCount()):
                    row = []
                    for c in cols:
                        it = self.table.item(r, c)
                        if it is not None:
                            row.append(it.text())
                        else:
                            w = self.table.cellWidget(r, c)
                            if isinstance(w, QLabel):
                                row.append(w.text())
                            else:
                                row.append("")
                    wr.writerow(row)

            QMessageBox.information(self, "내보내기 성공", f"저장됨:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "내보내기 실패", f"CSV 저장 중 오류:\n{e}")

    def _choose_export_columns(self) -> list[int] | None:
        """내보낼 컬럼을 체크박스로 선택하는 간단한 다이얼로그.
        반환: 체크된 컬럼 인덱스 리스트(순서는 원래 순서), 취소 시 None."""
        dlg = QDialog(self)
        dlg.setWindowTitle("내보낼 컬럼 선택")
        dlg.setModal(True)

        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("CSV로 내보낼 컬럼을 선택하세요.", dlg))

        lst = QListWidget(dlg)
        lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        # 현재 헤더를 기반으로 항목 생성 (▲/▼ 제거)
        for col in range(self.table.columnCount()):
            item = QListWidgetItem()
            hdr_it = self.table.horizontalHeaderItem(col)
            text = (hdr_it.text() if hdr_it else f"컬럼 {col}").replace(" ▲", "").replace(" ▼", "")
            item.setText(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)  # 기본: 전부 선택
            item.setData(Qt.ItemDataRole.UserRole, col)
            lst.addItem(item)
        v.addWidget(lst)

        # 버튼들
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dlg)
        v.addWidget(btns)

        # 선택/해제 단축 버튼(선택)
        hb = QHBoxLayout()
        sel_all = QPushButton("전체 선택", dlg)
        clr_all = QPushButton("전체 해제", dlg)
        hb.addWidget(sel_all)
        hb.addWidget(clr_all)
        v.insertLayout(2, hb)

        def _set_all(state: Qt.CheckState):
            for i in range(lst.count()):
                lst.item(i).setCheckState(state)

        sel_all.clicked.connect(lambda: _set_all(Qt.CheckState.Checked))
        clr_all.clicked.connect(lambda: _set_all(Qt.CheckState.Unchecked))
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        # 체크된 컬럼만 모아 반환 (원래 컬럼 순서 유지)
        chosen: list[int] = []
        for i in range(lst.count()):
            it = lst.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                chosen.append(int(it.data(Qt.ItemDataRole.UserRole)))
        return chosen


    def _load_all(self):
        """
        전체 데이터 조회 + 필터 적용.

        - UI 'Destination' 검색은 DB status 컬럼을 사용.
        - 날짜 필터는 테이블에서 실제 존재하는 첫 번째 후보 컬럼으로 적용.
        - KST(self.tz) 범위를 UTC 저장 DB에 그대로 넘김(필요시 변환 로직 추가).
        """
        if self._conn is None:
            return

        table = self.cfg.db.table_jobs

        # --- 테이블 컬럼 목록 로드 -------------------------------------------------
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {table} LIMIT 0")
                colnames = {d.name for d in cur.description}
        except Exception as e:
            QMessageBox.critical(self, "DB 오류", f"테이블 메타 조회 실패: {e}")
            return

        # 날짜 컬럼 후보 우선순위
        date_candidates = ("created_at", "updated_at", "timestamp", "created", "dt", "ts")
        date_col = next((c for c in date_candidates if c in colnames), None)
        if date_col is None:
            QMessageBox.critical(self, "DB 오류", "날짜 컬럼을 찾을 수 없습니다.")
            return

        # --- 날짜 범위(QDate → Python date) ---------------------------------------
        qd_from = self.date_from.date()
        qd_to = self.date_to.date()

        py_from = qdate_to_date(qd_from)
        py_to = qdate_to_date(qd_to)

        # 종료일은 '그날 끝'까지 포함하려면 다음날 0시 exclusive 범위 사용
        start_dt = datetime(py_from.year, py_from.month, py_from.day, tzinfo=self.tz)
        end_dt = datetime(py_to.year, py_to.month, py_to.day, tzinfo=self.tz) + timedelta(days=1)

        # --- 검색 필드 매핑 --------------------------------------------------------
        field_map = {}
        if "pid" in colnames:
            field_map["PID"] = "pid"
        if "sscc" in colnames:
            field_map["SSCC"] = "sscc"
        if "status" in colnames:
            field_map["Destination"] = "status"  # UI Destination → status
        if "barcode" in colnames:
            field_map["barcode"] = "barcode"
        if "error_reason" in colnames:
            field_map["error_reason"] = "error_reason"

        field_choice = self.search_field.currentText()
        kw = self.search_input.text().strip()

        # --- WHERE 구성 ------------------------------------------------------------
        where_parts = []
        params: List[Any] = []

        where_parts.append(f"{date_col} >= %s AND {date_col} < %s")
        params.extend([start_dt, end_dt])

        # 이번 검색에서 기본으로 사용할 컬럼 집합(콤보 선택 반영)
        if field_choice in field_map:
            search_cols = [field_map[field_choice]]
        else:
            # '전체'일 때 존재하는 컬럼만
            search_cols = [c for c in ("pid", "sscc", "status", "barcode", "error_reason") if c in colnames]

        # --- 키워드 WHERE 만들기 (다중조건/따옴표/OR/제외/특수문자/바코드 normalize) ---
        if kw:
            kw_parts, kw_params = _build_keyword_filters(kw, search_cols, colnames)
            if kw_parts:
                where_parts.append("(" + " AND ".join(kw_parts) + ")")
                params.extend(kw_params)

        # --- SELECT 컬럼 구성 ------------------------------------------------------
        select_cols = []
        if "id" in colnames:
            select_cols.append("id")
        if "pid" in colnames:
            select_cols.append("pid")
        if "sscc" in colnames:
            select_cols.append("sscc")
        if "status" in colnames:
            select_cols.append("status AS destination")
        if "barcode" in colnames:
            select_cols.append("barcode")
        if "error_reason" in colnames:
            select_cols.append("error_reason")
        else:
            select_cols.append("'' AS error_reason")
        if "image_path" in colnames:  # 이미지 경로 컬럼 추가
            select_cols.append("image_path")
        else:
            select_cols.append("'' AS image_path")
        select_cols.append(f"{date_col} AS dt")

        sql = f"SELECT {', '.join(select_cols)} FROM {table}"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        if "id" in colnames:
            sql += " ORDER BY id DESC"
        else:
            sql += f" ORDER BY {date_col} DESC"

        rows = self._exec_query(sql, tuple(params))
        self._populate(rows)

    def _populate(self, rows: List[Dict[str, Any]]):
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows, start=1):
            # 데이터 설정
            # items = [
            #     self._NumericItem(str(i)),
            #     QTableWidgetItem(str(row.get("pid", ""))),
            #     QTableWidgetItem(str(row.get("sscc", ""))),
            #     QTableWidgetItem(str(row.get("destination", ""))),
            #     QTableWidgetItem(str(row.get("barcode", ""))),
            #     QTableWidgetItem(str(row.get("error_reason", ""))),
            #     QTableWidgetItem(str(row.get("image_path", ""))),
            # ]
            items = [
                self._NumericItem(str(i)),
                QTableWidgetItem(str(row.get("pid", ""))),
                QTableWidgetItem(str(row.get("sscc", ""))),
                QTableWidgetItem(str(row.get("destination", ""))),
                QTableWidgetItem(str(row.get("barcode", ""))),
                QTableWidgetItem(str(row.get("error_reason", ""))),
                QTableWidgetItem(""),  # ★ 셀 위젯(라벨)만 보이게 아이템 텍스트는 비움
            ]

            # 처리 상태 색 지정 (아이템을 테이블에 넣기 전에)
            apply_status_cell_colors(items[3])

            # 날짜/시간 항목
            dt_val = row.get("dt")
            if isinstance(dt_val, datetime):
                dt_txt = dt_val.astimezone(self.tz).strftime("%Y-%m-%d %H:%M:%S")
            else:
                dt_txt = str(dt_val) if dt_val is not None else ""
            items.append(QTableWidgetItem(dt_txt))

            # 각 항목을 테이블에 추가하고 가운데 정렬 설정
            for col, item in enumerate(items):
                self.table.setItem(i - 1, col, item)
                ## ⛔ QSS로 아이템 정렬은 불가 → 코드에서 정렬 유지
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  # 텍스트 가운데 정렬

            ## 이미지 경로 컬럼에 툴팁 추가
            # if row.get("image_path"):
            #     img_item = self.table.item(i - 1, 6)
            #     img_item.setToolTip("더블클릭하여 이미지 열기")

            img_path = str(row.get("image_path", "") or "")
            if img_path:
                # 기존 툴팁(아이템에 추가)
                img_item = self.table.item(i - 1, 6)
                if img_item:
                    img_item.setToolTip("더블클릭하여 이미지 열기")
                    img_item.setText("")  # ★ 혹시 모를 겹침 방지(명시적으로 비우기)

                # ✅ 라벨을 셀 위젯으로 올려 색상을 QSS로 제어
                lbl = QLabel(img_path)
                lbl.setObjectName("ImagePathLabel")
                lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
                lbl.setToolTip("더블클릭하여 이미지 열기")
                # 파일 존재 여부를 동적 프로퍼티로 표시 → QSS에서 색상 분기
                exists = os.path.exists(img_path)
                lbl.setProperty("fileExists", "true" if exists else "false")
                # 즉시 반영(동적 프로퍼티 적용 강제)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

                self.table.setCellWidget(i - 1, 6, lbl)

        # ✅ 루프가 끝난 뒤 '한 번만' 정렬/표시 유지
        if getattr(self, "_sort_col", -1) in getattr(self, "_allowed_sort_cols", set()):
            hdr = self.table.horizontalHeader()
            hdr.setSortIndicator(self._sort_col, self._sort_order)
            self.table.setSortingEnabled(True)
            self.table.sortItems(self._sort_col, self._sort_order)
            self.table.setSortingEnabled(False)

        # ▲/▼ 헤더 텍스트 갱신
        self._update_header_sort_icons()



                # 이미지 존재 여부에 따라 색상 설정
                ## 이런 하드코딩된 색상 코드('#1d6f42', '#d13438')들을 QSS로 분리하고,
                # 코드에서는 상태만 지정하는 방식(예: setProperty("fileExists", True))으로 리팩토링할 수 있습니다.
                # if os.path.exists(str(row.get("image_path", ""))):
                #     img_item.setForeground(QColor("#1d6f42"))  # 초록색 - 파일 존재
                # else:
                #     img_item.setForeground(QColor("#d13438"))  # 빨간색 - 파일 없음

# ---------------------------------------------------------------------------
# 메인 윈도우 구현
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, cfg: AppConfig, tz: ZoneInfo, parent=None): ## 전역 설정 및 인라인 스트립 설정
        super().__init__(parent)
        self.cfg = cfg
        self.tz = tz
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        print("data_table objectName =", self.ui.data_table.objectName())
        self._pack_m1panel()  ## ⬅️ 추가: 패널 재구성
        QTimer.singleShot(0, self._apply_logo)
        self._init_indicator_icons()
        self._shrink_image_box(420, 320)
        QTimer.singleShot(0, self._place_conn_log_below_image)
        QTimer.singleShot(0, self._realign_right_panel_to_image)

        # Qt의 QSS가 테이블 셀 배경을 항상 확실히 먹이지 못하는 케이스가 있어 강제로 수동 설정
        p = self.ui.data_table.palette()
        p.setColor(QPalette.ColorRole.Base, QColor("#2C2F36"))
        p.setColor(QPalette.ColorRole.AlternateBase, QColor("#34383F"))
        p.setColor(QPalette.ColorRole.Text, QColor("#E0E0E0"))
        p.setColor(QPalette.ColorRole.Highlight, QColor("#0D6EFD"))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        self.ui.data_table.setPalette(p)
        self.ui.data_table.setAlternatingRowColors(True)
        self.ui.data_table.setShowGrid(True)

        ## ★ 자동생성 파일의 인라인 스타일 제거(전역 QSS가 우선 적용되게)
        #    특정 위젯은 유지하고 싶다면 keep 집합에 objectName을 추가하세요.
        #    예: keep={"data_table"}
        strip_inline_styles(self)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )

        # 센터명 적용
        self.ui.center_name.setText(self.cfg.center_name)
        # 센터명 깜빡임 타이머
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_center_name_style)
        self._blink_state = False

        # 상태 텍스트 초기화
        self._running = False
        self._update_run_state(False, initial=True)

        # 버튼 연결
        self.ui.pushButton.clicked.connect(self._toggle_run_clicked)
        self.ui.pushButton_2.clicked.connect(self.close)
        self.ui.pushButton_3.clicked.connect(self.showMinimized)
        self.ui.pushButton_4.clicked.connect(self._open_all_data_dialog)

        # 테이블 기본 설정
        self._init_table()
        # (번갈이 행 배경을 쓰려면 _init_table 안에서 setAlternatingRowColors(True)가 켜져 있어야 함)
        # 중복 호출해도 문제는 없음:
        # self.ui.data_table.setAlternatingRowColors(True)
        # 중앙정렬할 컬럼들: no(0), pid(1), sscc(2), 처리 상태(4), error_reason(5)
        for col in (0, 1, 2, 5):
            self.ui.data_table.setItemDelegateForColumn(col,
                                                        AlignDelegate(Qt.AlignmentFlag.AlignCenter, self.ui.data_table))
        # 처리 상태(4번 컬럼)만 배경색 델리게이트 적용
        self.ui.data_table.setItemDelegateForColumn(4, StatusBGDelegate(self.ui.data_table))

        # 실시간 시계
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()  # 즉시 1회

        # 이미지 폴링
        self._image_timer = QTimer(self)
        self._image_timer.timeout.connect(self._update_latest_image)
        self._image_timer.start(int(self.cfg.image_poll_interval_sec * 1000))
        self._last_image_path: Optional[pathlib.Path] = None

        # DB 폴링 스레드 시작
        self._db_thread = DBPollerThread(self.cfg, self.tz, self)
        self._db_thread.daily_count_signal.connect(self._on_daily_count)
        self._db_thread.latest_rows_signal.connect(self._on_latest_rows)
        self._db_thread.db_error_signal.connect(self._on_db_error)
        self._db_thread.start()

        # 인디케이터 UI 초기값
        self.set_network_status(False)
        self.set_camera_status(False)
        self.set_plc_status(False)

        # 연결 상태 모니터링
        self._setup_heartbeat_monitor()  # plc, cam
        self._setup_network_monitor()    # 네트워크
        self._setup_speed_monitor()      # 속도
        self._setup_connection_log()

        QTimer.singleShot(0, self._place_conn_log_below_image)

    def _pack_m1panel(self):
        p = self.ui.m1panel
        lay = p.layout()
        if lay is None:
            lay = QFormLayout(p)

        # 기존 레이아웃 비우기
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                sub = item.layout()
                while sub.count():
                    si = sub.takeAt(0)
                    if si.widget():
                        si.widget().setParent(None)

        # 인라인 스타일 제거(패널/라벨 모두 QSS에 맡김)
        for w in (self.ui.center_name, self.ui.count_title, self.ui.count_label, self.ui.line_1):
            w.setStyleSheet("")
        p.setStyleSheet("")

        # 패널 내부 컨테이너 + 레이아웃
        row = QWidget(p)
        h = QHBoxLayout(row)
        h.setContentsMargins(12, 8, 12, 12)  # ⬅️ 바닥 여유 조금 더
        h.setSpacing(8)

        # 왼쪽: 센터명
        left = self.ui.center_name
        left.setParent(row)
        left.setWordWrap(True)
        left.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        h.addWidget(left, 2)

        # 오른쪽: 제목 / 구분선 / 수량
        right_box = QWidget(row)
        v = QVBoxLayout(right_box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        # 제목
        title = self.ui.count_title
        title.setParent(right_box)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)

        # 구분선(라벨로 생성되어 있어도 선처럼 보이게 보정)
        sep = self.ui.line_1
        sep.setParent(right_box)
        try:
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            sep.setFixedHeight(2)
        except Exception:
            pass
        v.addWidget(sep)

        # 수량(여기서 잘림 방지 처리)
        num = self.ui.count_label
        num.setParent(right_box)
        num.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        fm = QFontMetrics(num.font())
        num.setMinimumHeight(max(68, fm.height() + 10))  # ⬅️ 폰트 기준 최소 높이
        num.setContentsMargins(0, 0, 0, 2)  # ⬅️ 아주 살짝 아래 여유
        num.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        v.addWidget(num)

        h.addWidget(right_box, 1)
        lay.addRow(row)

    # ------------------------------------------------------------------
    # UI 초기화 도우미
    # ------------------------------------------------------------------
    def _init_table(self):
        """
        data_table 위젯의 초기 설정을 수행합니다.
        - 컬럼명과 너비를 설정합니다.
        - 테이블과 헤더의 스타일을 지정합니다.
        """
        # 0) QSS 스코프용 objectName (style.css의 #DataTable 규칙이 이걸 타깃팅)
        self.ui.data_table.setObjectName("DataTable")
        # ✅ 즉시 스타일 재적용 (선택이지만 권장)
        self.ui.data_table.style().unpolish(self.ui.data_table)
        self.ui.data_table.style().polish(self.ui.data_table)

        # 1. 컬럼명 설정
        # 원하시는 컬럼명으로 수정하세요.
        headers = ["no.", "pid",  "sscc", "barcode", "처리 상태", "error_reason"]
        self.ui.data_table.setColumnCount(len(headers))
        self.ui.data_table.setHorizontalHeaderLabels(headers)

        # 2. 컬럼 크기(너비) 조정  (QSS로 불가 → 코드 유지) ##
        self.ui.data_table.setColumnWidth(0, 90)  # no
        self.ui.data_table.setColumnWidth(1, 90)  # pid
        self.ui.data_table.setColumnWidth(2, 80)  # sscc
        self.ui.data_table.setColumnWidth(3, 130)  # barcode
        self.ui.data_table.setColumnWidth(4, 20)  # 처리 상태
        self.ui.data_table.setColumnWidth(5, 260)  # error_reason
        # 마지막 '상세 내용' 컬럼은 창 크기에 맞춰 남은 공간을 모두 차지하도록 설정
        self.ui.data_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        # 3. 테이블/헤더 전반 스타일
        ##    색/패딩/선택색 등은 전역 style.css에서 관리.
        #    번갈이 색 활성화는 코드에서 ON 필요.
        self.ui.data_table.setAlternatingRowColors(True)

        ## ⛔ 인라인 스타일은 전역 QSS로 이동 → 주석/삭제
        # style_sheet = """
        #     QTableWidget {
        #         background-color: #CCCCCC; /* 테이블 전체 배경색 */
        #         alternate-background-color: #E0E0E0; /* 번갈아 나오는 행 배경색 */
        #         gridline-color: #DDE2E6; /* 그리드 라인 색상 */
        #         color: #000000; /* 기본 글자색 */
        #         border: 1px solid #CED4DA;
        #         font-size: 13px;
        #     }
        #     /* 테이블 헤더 스타일 */
        #     QHeaderView::section {
        #         background-color: #495057; /* 헤더 배경색 (어두운 회색) */
        #         color: #FFFFFF; /* 헤더 글자색 (흰색) */
        #         font-weight: bold;
        #         padding: 6px;
        #         border: none;
        #         border-bottom: 1px solid #343A40;
        #     }
        #     /* 테이블 셀 스타일 */
        #     QTableWidget::item {
        #         border-bottom: 1px solid #E9ECEF;
        #         padding: 5px;
        #     }
        #     /* 사용자가 셀을 선택했을 때 스타일 */
        #     QTableWidget::item:selected {
        #         background-color: #0D6EFD; /* 선택된 아이템 배경색 (파란색) */
        #         color: #FFFFFF; /* 선택된 아이템 글자색 (흰색) */
        #     }
        # """
        # self.ui.data_table.setStyleSheet(style_sheet)

        # 4. 동작 설정 (스타일 아님 → 코드에 유지) ##
        self.ui.data_table.setShowGrid(True)  # 셀 사이의 그리드 라인 표시
        self.ui.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # 테이블 내용 직접 편집 방지
        self.ui.data_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # 클릭 시 행 전체 선택
        self.ui.data_table.verticalHeader().setVisible(False)  # 수직 헤더(행 번호) 숨기기

    # ------------------------------------------------------------------
    # 로고 이미지
    # ------------------------------------------------------------------
    def _apply_logo(self):
        lbl = getattr(self.ui, "logo_1", None) or self.findChild(QLabel, "logo_1")
        if not lbl:
            return

        path = resource_path("ui", "resources", "IDT_logo.png")
        pm = QPixmap(path)
        if pm.isNull():
            lbl.setText(f"로고 파일 없음\n{path}")
            return

        lbl.setText("")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("border: none; background: transparent;")

        # 원본 보관 + 리사이즈시 재스케일을 위해 이벤트 필터 설치
        self._logo_pixmap = pm
        lbl.installEventFilter(self)

        # 최초 1회 스케일링
        self._rescale_logo(lbl)

    def _screen_dpr(self) -> float:
        try:
            wh = self.windowHandle() or self.window().windowHandle()
            if wh and wh.screen():
                # Qt6에선 float, 일부 환경에선 1/1.25/1.5/2 등
                return float(wh.screen().devicePixelRatio())
        except Exception:
            pass
        return 1.0

    def _rescale_logo(self, lbl: QLabel):
        if not hasattr(self, "_logo_pixmap"):
            return
        pm = self._logo_pixmap
        if pm.isNull():
            return

        dpr = self._screen_dpr()
        # 라벨의 '논리' 크기에 DPR을 곱해 '물리' 픽셀 단위로 스케일
        tw = max(1, int(lbl.width() * dpr))
        th = max(1, int(lbl.height() * dpr))

        scaled = pm.scaled(tw, th,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        # 이 픽스맵은 dpr 스케일이라는 걸 Qt에 알려주기
        scaled.setDevicePixelRatio(dpr)
        lbl.setPixmap(scaled)

    def eventFilter(self, obj, ev):
        # 로고 라벨 리사이즈 때마다 다시 스케일
        if isinstance(obj, QLabel) and obj.objectName() == "logo_1":
            if ev.type() == ev.Type.Resize:
                self._rescale_logo(obj)
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------
    # 시계 업데이트
    # ------------------------------------------------------------------
    @Slot()
    def _update_clock(self):
        now_txt = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S")
        self.ui.datetime_label.setText(now_txt)

    # ------------------------------------------------------------------
    # 이미지 갱신
    # ------------------------------------------------------------------
    @Slot()
    def _update_latest_image(self):
        root = self.cfg.paths.mapped_image_root
        today_str = datetime.now(self.tz).strftime("%Y%m%d")
        day_dir = root / today_str
        if not day_dir.is_dir():
            # 경로 없음 → 검정 배경 유지
            return
        # 이미지 확장자 후보
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(str(day_dir / pat)))
        if not files:
            return
        latest_path = max(files, key=os.path.getmtime)
        if self._last_image_path and str(self._last_image_path) == latest_path:
            return  # 변화 없음
        self._last_image_path = pathlib.Path(latest_path)
        pix = QPixmap(latest_path)
        if not pix.isNull():
            # 라벨 크기에 맞춰 유지비율 스케일
            lbl = self.ui.image_label
            scaled = pix.scaled(lbl.width(), lbl.height(), Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(scaled)

    def _toggle_center_name_style(self):
        """센터명 레이블 스타일을 토글하여 깜빡임 효과 생성"""
        self._blink_state = not self._blink_state
        ## ⛔ 인라인 스타일 제거(전역 QSS가 적용되도록)
        # if self._blink_state:
        #     # 강조 스타일 (깜빡일 때)
        #     self.ui.center_name.setStyleSheet(
        #         u"font-size:42px; color:#F0F0F0; background-color: #F0F0F0; font-weight:bold;")
        # else:
        #     # 기본 스타일 (원래 상태)
        #     self.ui.center_name.setStyleSheet(
        #         u"font-size:42px; color:#888; background-color: #F0F0F0; font-weight:bold;")

        # ✅ QSS가 읽을 동적 프로퍼티만 토글
        self.ui.center_name.setProperty("blinking", self._blink_state)
        # 즉시 반영
        self.ui.center_name.style().unpolish(self.ui.center_name)
        self.ui.center_name.style().polish(self.ui.center_name)

    # ------------------------------------------------------------------
    # DB콜백
    # ------------------------------------------------------------------
    @Slot(int)
    def _on_daily_count(self, count: int):
        self.ui.count_label.setText(f"{count:,}건")

    # 일반 작업 테이블 구성
    @Slot(list)
    def _on_latest_rows(self, rows: List[Dict[str, Any]]):
        tbl = self.ui.data_table
        tbl.setRowCount(len(rows))
        for idx, row in enumerate(rows, start=1):
            # 번호 컬럼: 최근건이 1번
            no_item = QTableWidgetItem(str(row.get("id", "")))
            pid_item = QTableWidgetItem(str(row.get("pid", "")))
            sscc_item = QTableWidgetItem(str(row.get("sscc", "")))
            barcode_item = QTableWidgetItem(str(row.get("barcode", "")))
            dest_item = QTableWidgetItem(str(row.get("destination", "")))
            error_reason = QTableWidgetItem(str(row.get("error_reason", "")))

            apply_status_cell_colors(dest_item)

            ## [BEFORE] 개별 아이템마다 중앙 정렬
            # for it in (no_item, pid_item, sscc_item, dest_item, err_item):
            #     it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # [NOW] 정렬은 컬럼 delegate(AlignDelegate)가 담당 → 위 루프 불필요
            #       (필요 시 barcode(3)도 delegate 설치로 중앙정렬 가능)

            tbl.setItem(idx - 1, 0, no_item)
            tbl.setItem(idx - 1, 1, pid_item)
            tbl.setItem(idx - 1, 2, sscc_item)
            tbl.setItem(idx - 1, 3, barcode_item)
            tbl.setItem(idx - 1, 4, dest_item)
            tbl.setItem(idx - 1, 5, error_reason)

    @Slot(str)
    def _on_db_error(self, msg: str):
        # 상태바에만 표시 (팝업 자제)
        self.statusBar().showMessage(msg, 5000)

    # ------------------------------------------------------------------
    # 시작/중단 버튼 제어
    # ------------------------------------------------------------------
    def _toggle_run_clicked(self):
        self._running = not self._running
        self._update_run_state(self._running)

        command = "start" if self._running else "stop"

        try:
            # 백엔드(main.py)의 command_server와 통신합니다.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('127.0.0.1', 15005))  # main.py에 설정한 포트와 동일해야 합니다.
                s.sendall(command.encode('utf-8'))
                response = s.recv(1024)
                print(f"Backend response: {response.decode('utf-8')}")

        except ConnectionRefusedError:
            print("백엔드에 연결할 수 없습니다. 백엔드 프로세스가 실행 중인지 확인하세요.")
            # 사용자에게 상태바 등을 통해 오류 메시지를 표시할 수 있습니다.
            # 예: self.statusBar().showMessage("백엔드 연결 실패", 5000)

            # 통신에 실패했으므로 UI 상태를 이전으로 되돌립니다.
            self._running = not self._running
            self._update_run_state(self._running)

        except Exception as e:
            print(f"명령 전송 중 오류 발생: {e}")

    # ------------------------------------------------------------------
    # 작업 시작/중단 시 center_name 점멸 _toggle_center_name_style 참고
    # ------------------------------------------------------------------
    def _update_run_state(self, running: bool, *, initial: bool = False):
        self._running = running
        if running:
            self.ui.pushButton.setText("작업 중단")

            ## [BEFORE] 인라인 스타일로 '작업 중단' 버튼 외형 지정
            # self.ui.pushButton.setStyleSheet(
            #     "font-size:30px;color:#ffffff;font-weight:bold; background:#ff4c4c;  border-radius:12px; border:2px solid #000000;"
            # )

            # [NOW] 상태만 토글 → QSS가 처리
            self.ui.pushButton.setProperty("running", True)
            self.ui.pushButton.style().unpolish(self.ui.pushButton)
            self.ui.pushButton.style().polish(self.ui.pushButton)

            # 점멸 시작
            self._blink_timer.start(700)

        else:
            self.ui.pushButton.setText("작업 시작")

            ## [BEFORE] 인라인 스타일로 '작업 시작' 버튼 외형 지정
            # self.ui.pushButton.setStyleSheet(
            #     "font-size:30px;color:#0c61c4;font-weight:bold; background:#c6dff5;  border-radius:12px; border:2px solid #000000;"
            # )

            # [NOW] 상태만 토글 → QSS가 처리 (기본 상태로 복귀)
            self.ui.pushButton.setProperty("running", False)
            self.ui.pushButton.style().unpolish(self.ui.pushButton)
            self.ui.pushButton.style().polish(self.ui.pushButton)

            # 점멸 정지 + 라벨을 기본상태로 돌림(QSS가 기본 스타일 적용)
            self._blink_timer.stop()
            self.ui.center_name.setProperty("blinking", False)
            self.ui.center_name.style().unpolish(self.ui.center_name)
            self.ui.center_name.style().polish(self.ui.center_name)

        ## (선택) 상태 텍스트 갱신이 필요하면 여기에:
        # if not initial:
        #     self.ui.count_title_2.setText("작업 진행 중" if running else "시작 대기 중")

    # ------------------------------------------------------------------
    # 전체 데이터 팝업 열기
    # ------------------------------------------------------------------
    @Slot()
    def _open_all_data_dialog(self):
        dlg = AllDataDialog(self.cfg, self.tz, self)
        dlg.exec()

    # ------------------------------------------------------------------
    # 인디케이터 표시 (네트워크/카메라/PLC)
    # ------------------------------------------------------------------
    def _setup_network_monitor(self):
        """
        QTcpSocket을 사용하여 비동기적으로 네트워크 상태를 확인하도록 설정합니다.
        UI 멈춤 현상이 발생하지 않습니다.
        """
        # 1. 소켓 및 타이머 객체 생성
        self._network_socket = QTcpSocket(self)
        self._network_timer = QTimer(self)
        self._network_timer.setInterval(5000)  # 5초 간격

        # 2. 시그널(Signal)과 슬롯(Slot) 연결
        #    연결 성공, 실패(오류 발생) 시 각각의 메서드가 호출됩니다.
        self._network_socket.connected.connect(self._on_network_connected)
        self._network_socket.errorOccurred.connect(self._on_network_error)

        #    타이머의 timeout 시그널을 네트워크 확인 시작 메서드에 연결합니다.
        self._network_timer.timeout.connect(self._check_network_status)

        # 3. 타이머 시작 및 즉시 확인
        self._network_timer.start()
        self._check_network_status()

    def _check_network_status(self):
        """
        Google DNS 서버로 비동기 연결을 시도합니다.
        이 함수는 즉시 반환되며, 결과는 시그널을 통해 처리됩니다.
        """
        self._network_socket.connectToHost(QHostAddress("8.8.8.8"), 53)

    def _on_network_connected(self):
        """네트워크 연결에 성공했을 때 호출되는 슬롯입니다."""
        self.set_network_status(True)
        # 확인 후 바로 연결을 끊습니다. 계속 연결을 유지할 필요가 없습니다.
        self._network_socket.abort()

    def _on_network_error(self, socket_error):
        """네트워크 연결에 실패했을 때 호출되는 슬롯입니다."""
        # 연결 실패 시 소켓 상태를 초기화합니다.
        self._network_socket.abort()
        self.set_network_status(False)
        # 필요하다면 오류 로그를 출력할 수 있습니다.
        # print(f"Network check failed: {self._network_socket.errorString()}")

    def set_network_status(self, ok: bool):
        """네트워크 상태를 설정하고 변경이 있으면 로그에 기록합니다."""
        # 이전 상태 저장 (속성이 없으면 None으로 초기화)
        prev_status = getattr(self, '_prev_network_status', None)

        # 상태가 변경된 경우에만 로그 추가
        if prev_status != ok:
            # 현재 시간 포맷
            timestamp = datetime.now().strftime('%H:%M:%S')

            # 로그 메시지 생성
            log_msg = f"{timestamp} - 네트워크: {'연결됨' if ok else '끊어짐'}"

            # 로그 목록에 추가
            if hasattr(self, 'connection_logs'):
                self.connection_logs.append(log_msg)
                # 최대 12개 항목만 유지
                if len(self.connection_logs) > 12:
                    self.connection_logs.pop(0)

                # logo_3 위젯 업데이트
                self.ui.conn_log.setText("연결 상태 로그:\n" + "\n".join(self.connection_logs))

            # 현재 상태 저장
            self._prev_network_status = ok

        # 인디케이터 업데이트
        self._set_indicator_icon(self.ui.logo_label_2, ok, "net")


    def _setup_heartbeat_monitor(self):
        """하트비트 상태를 주기적으로 확인할 타이머를 설정하고 시작합니다."""
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._check_heartbeat_status)
        # 1초마다 상태를 확인하여 UI에 즉시 반영합니다.
        self.heartbeat_timer.start(1000)
        # 프로그램 시작 시 즉시 한 번 실행
        self._check_heartbeat_status()

    def _check_heartbeat_status(self):
        """
        heartbeat_status.json 파일을 읽어 UI의 연결 상태 표시등을 업데이트합니다.
        마지막 신호 수신 시간이 10초 이내일 경우 '정상'으로 간주합니다.
        """
        try:
            # heartbeat_status.json 파일의 경로를 설정합니다. (필요시 경로 수정)
            status_file_path = Path("C:/Users/USER/Desktop/1DT/heartbeat_status.json")

            if not status_file_path.exists():
                # 파일이 없으면 두 상태 모두 '연결 끊김'으로 처리
                self.set_camera_status(False)
                self.set_plc_status(False)
                return

            with open(status_file_path, 'r', encoding='utf-8') as f:
                heartbeat_data = json.load(f)

            now = datetime.now()
            connection_timeout = timedelta(seconds=10)

            # 카메라 상태 확인
            camera_info = heartbeat_data.get('camera', {})
            camera_last_seen_str = camera_info.get('last_seen')
            is_camera_ok = False
            if camera_last_seen_str:
                last_seen_dt = datetime.strptime(camera_last_seen_str, '%Y-%m-%d %H:%M:%S')
                if (now - last_seen_dt) <= connection_timeout:
                    is_camera_ok = True
            self.set_camera_status(is_camera_ok)

            # PLC 상태 확인 (JSON 파일의 키가 'PLC'이므로 주의)
            plc_info = heartbeat_data.get('PLC', {})
            plc_last_seen_str = plc_info.get('last_seen')
            is_plc_ok = False
            if plc_last_seen_str:
                last_seen_dt = datetime.strptime(plc_last_seen_str, '%Y-%m-%d %H:%M:%S')
                if (now - last_seen_dt) <= connection_timeout:
                    is_plc_ok = True
            self.set_plc_status(is_plc_ok)

        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            # 파일 읽기 오류 발생 시 모든 연결 상태를 '끊김'으로 처리
            self.set_camera_status(False)
            self.set_plc_status(False)
            # 필요하다면 여기에 오류 로깅 코드를 추가할 수 있습니다.
            # print(f"하트비트 상태 확인 중 오류 발생: {e}")

    def set_camera_status(self, ok: bool):
        """카메라 상태를 설정하고 변경이 있으면 로그에 기록합니다."""
        prev_status = getattr(self, "_prev_camera_status", "INIT")

        # bool → "GREEN" / None → "ORANGE" / False → "RED"
        status_key = (
            "GREEN" if ok is True else "RED" if ok is False else "ORANGE"
        )

        if prev_status != status_key:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_msg = (
                f"{timestamp} - 카메라: "
                f"{'모두 정상' if ok is True else '모두 끊어짐' if ok is False else '일부 이상'}"
            )

            if hasattr(self, "connection_logs"):
                self.connection_logs.append(log_msg)
                if len(self.connection_logs) > 12:
                    self.connection_logs.pop(0)
                self.ui.conn_log.setText(
                    "연결 상태 로그:\n" + "\n".join(self.connection_logs)
                )

            self._prev_camera_status = status_key

        # 인디케이터 갱신
        self._set_indicator_icon(self.ui.hm_logo_2, ok, "cam")

    def set_plc_status(self, ok: bool):
        """PLC 상태를 설정하고 변경이 있으면 로그에 기록합니다."""
        # 이전 상태 저장 (속성이 없으면 None으로 초기화)
        prev_status = getattr(self, '_prev_plc_status', None)
        # 상태가 변경된 경우에만 로그 추가
        if prev_status != ok:
            # 현재 시간 포맷
            timestamp = datetime.now().strftime('%H:%M:%S')

            # 로그 메시지 생성
            log_msg = f"{timestamp} - PLC: {'연결됨' if ok else '끊어짐'}"

            # 로그 목록에 추가
            if hasattr(self, 'connection_logs'):
                self.connection_logs.append(log_msg)
                # 최대 12개 항목만 유지
                if len(self.connection_logs) > 12:
                    self.connection_logs.pop(0)

                # logo_3 위젯 업데이트
                self.ui.conn_log.setText("연결 상태 로그:\n" + "\n".join(self.connection_logs))

            # 현재 상태 저장
            self._prev_plc_status = ok

        # 인디케이터 업데이트
        self._set_indicator_icon(self.ui.datetime_label_2, ok, "plc")

    def _apply_indicator(self, widget: QLabel, ok: bool, text: str):
        # 텍스트/HTML 흔적 제거
        widget.clear()
        widget.setText("")
        widget.setTextFormat(Qt.TextFormat.PlainText)

        widget.setProperty("indicator", True)
        widget.setProperty("status", "ok" if ok else "error")
        widget.setStyleSheet("border:none; background:transparent;")
        widget.setMouseTracking(False)
        widget.setAttribute(Qt.WidgetAttribute.WA_Hover, False)

        # 아이콘 크기 키우기
        icon_px = 36
        widget.setFixedSize(icon_px, icon_px)
        widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 대상별 아이콘 경로
        name = widget.objectName() or ""
        if "logo_label_2" in name or text == "네트워크":
            on_icon = resource_path("ui", "resources", "network_conn_icon.png")
            off_icon = resource_path("ui", "resources", "network_conn_off_icon.png")
            tip = f"네트워크: {'연결' if ok else '끊김'}"
        elif "hm_logo_2" in name or text == "CAM":
            on_icon = resource_path("ui", "resources", "cam_on.png")
            off_icon = resource_path("ui", "resources", "cam_off.png")
            tip = f"카메라: {'연결' if ok else '끊김'}"
        elif "datetime_label_2" in name or text == "PLC":
            on_icon = resource_path("ui", "resources", "plc_on.png")
            off_icon = resource_path("ui", "resources", "plc_off.png")
            tip = f"PLC: {'연결' if ok else '끊김'}"
        else:
            on_icon = resource_path("ui", "resources", "network_conn_icon.png")
            off_icon = resource_path("ui", "resources", "network_conn_off_icon.png")
            tip = f"{text}: {'연결' if ok else '끊김'}"

        pm = QPixmap(on_icon if ok else off_icon)
        if pm.isNull():
            widget.setToolTip(f"{tip} (아이콘 없음)")
            widget.clear()
            return

        widget.setPixmap(pm.scaled(
            icon_px, icon_px,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        widget.setToolTip(tip)

    def _init_indicator_icons(self):
        """인디케이터 라벨을 아이콘-only로 초기화(텍스트 제거, 고정 크기, 투명 배경)."""
        ICON_PX = 32  # 원하면 36으로만 바꿔도 또렷함 체감↑
        for name in ("logo_label_2", "hm_logo_2", "datetime_label_2"):
            lbl = getattr(self.ui, name, None)
            if not lbl:
                continue
            lbl.setText("")  # 텍스트 제거
            lbl.setFixedSize(ICON_PX, ICON_PX)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("border: none; background: transparent;")
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lbl.setScaledContents(False)  # 라벨 내부에서 재스케일 금지

    # 연결 상태 변경 로그를 관리하기 위한 간단한 리스트
    def _setup_connection_log(self):
        """
        연결 상태 로그 설정
        """
        # 로그 항목을 저장할 리스트 초기화
        self.connection_logs = []

        self.ui.conn_log.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.ui.conn_log.setWordWrap(True)
        self.ui.conn_log.setText("연결 상태 로그:")
        self.ui.conn_log.raise_()  # ← 겹침 방지

    def _set_indicator_icon(self, label: QLabel, ok: bool | None, kind: str):
        """
            kind: 'net' | 'cam' | 'plc'
            ok:   True(정상)/False(오류)/None(일부 정상; cam만 사용)
            """
        label.setText("")  # 텍스트 절대 출력 안 함

        # 파일명 매핑
        if kind == "net":
            on_png = "network_conn_icon.png"
            off_png = "network_conn_off_icon.png"
            warn_png = off_png
            tip_title = "네트워크"
        elif kind == "cam":
            on_png = "cam_on.png"
            off_png = "cam_off.png"
            warn_png = "cam_partial.png"
            tip_title = "카메라"
        else:  # plc
            on_png = "plc_on.png"
            off_png = "plc_off.png"
            warn_png = off_png
            tip_title = "PLC"

        filename = on_png if ok is True else warn_png if ok is None else off_png
        path = resource_path("ui", "resources", filename)

        pm_src = QPixmap(path)
        if pm_src.isNull():
            label.clear()
            label.setToolTip(f"{tip_title}: {'연결' if ok else '끊김' if ok is False else '일부 이상'} (아이콘 없음)")
            return

        # 고DPI 고려: 라벨 픽셀 크기 × 장치 배율로 '한 번만' 스케일
        dpr = getattr(label, "devicePixelRatioF", lambda: 1.0)()
        tw = max(1, int(label.width() * dpr))
        th = max(1, int(label.height() * dpr))

        img = pm_src.toImage().scaled(
            tw, th,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation  # 작은 아이콘엔 Fast가 또렷함
        )
        pm = QPixmap.fromImage(img)
        pm.setDevicePixelRatio(dpr)

        label.setPixmap(pm)
        label.setToolTip(f"{tip_title}: {'연결' if ok else '끊김' if ok is False else '일부 이상'}")

    def _shrink_image_box(self, w: int, h: int):
        lbl = self.ui.image_label
        lbl.setMinimumSize(0, 0)  # .ui의 최소 크기(400x300) 해제
        lbl.setMaximumSize(w, h)
        lbl.setFixedSize(w, h)  # 딱 고정하고 싶으면
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _place_conn_log_below_image(self):
        """image_label 바로 아래에 conn_log를 고정 마진으로 배치(타이트 버전)."""
        log = getattr(self.ui, "conn_log", None)
        img = getattr(self.ui, "image_label", None)
        cw = getattr(self.ui, "centralwidget", None)
        if not (log and img and cw):
            return

        GAP = 6  # 이미지 아래 여백(줄임)
        DESIRED_H = 120  # 로그 상자 기본 높이(줄임)
        MIN_H = 80  # 최소 높이
        SIDE_MARGIN = 0  # 좌측을 이미지에 딱 맞춤
        MARGIN_BOTTOM = 8  # 하단 마진(줄임)

        left = img.x() + SIDE_MARGIN
        top = img.y() + img.height() + GAP
        width = img.width()

        max_h = max(0, cw.height() - top - MARGIN_BOTTOM)
        height = min(DESIRED_H, max_h)
        if height < MIN_H:
            height = max(0, max_h)
            if height < 30:
                log.hide()
                return
        log.show()
        log.setGeometry(QRect(left, top, width, height))
        log.raise_()

    def _realign_right_panel_to_image(self, gap_x: int = 12, right_margin: int = 12):
        """인식 이미지(label) 오른쪽에 '작업 현황' 타이틀/테이블을 붙여 정렬."""
        cw = getattr(self.ui, "centralwidget", None)
        img = getattr(self.ui, "image_label", None)
        tbl = getattr(self.ui, "data_table", None)
        title = getattr(self.ui, "image_title_2", None)
        if not (cw and img and tbl and title):
            return

        new_left = img.x() + img.width() + gap_x
        avail_w = max(0, cw.width() - new_left - right_margin)

        # 타이틀(작업 현황) 위치/폭 조정
        title_h = title.height()
        title.setGeometry(QRect(new_left, title.y(), avail_w, title_h))

        # 테이블 위치/폭 조정 (세로 크기는 그대로 유지)
        tbl.setGeometry(QRect(new_left, tbl.y(), avail_w, tbl.height()))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_conn_log_below_image()
        self._realign_right_panel_to_image()

    # ------------------------------------------------------------------
    # 속도 표시
    # ------------------------------------------------------------------

    def _setup_speed_monitor(self):
        """속도 정보 UI 업데이트를 위한 타이머를 설정하고 시작합니다."""
        self.speed_timer = QTimer(self)
        self.speed_timer.timeout.connect(self._update_speed_display)
        # 1000ms = 1초마다 timeout 시그널 발생
        self.speed_timer.start(1000)
        # UI가 처음 뜰 때 한번 바로 업데이트
        self._update_speed_display()

    def _update_speed_display(self):
        """
        speed_status.json 파일을 읽어 UI의 속도계를 업데이트합니다.
        속도(mm/s)를 cm/s로 변환하여 표시합니다.
        """
        try:
            # speed_status.json 파일의 경로를 설정합니다.
            speed_file_path = Path("C:/Users/USER/Desktop/1DT/speed_status.json")

            if not speed_file_path.exists():
                self.ui.speed_label.setText("0.0 cm/s")
                return

            with open(speed_file_path, 'r', encoding='utf-8') as f:
                speed_data = json.load(f)

            # JSON에서 'speed_mms' 값을 읽고 cm/s로 변환
            speed_mms = speed_data.get('speed_mms', 0.0)
            speed_cms = speed_mms / 10.0

            # 라벨 텍스트를 "0.0 cm/s" 형식으로 업데이트
            display_text = f"{speed_cms:.1f} cm/s"
            self.ui.speed_label.setText(display_text)

        except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
            # 오류 발생 시 안전하게 0으로 표시
            self.ui.speed_label.setText("0.0 cm/s")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """
        창이 닫힐 때 호출되는 이벤트 핸들러입니다.
        종료 여부를 묻는 확인 대화상자를 표시하고 모든 리소스를 안전하게 해제합니다.
        """
        # 1) QMessageBox 인스턴스 생성
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("종료 확인")
        msg_box.setText("종료하시겠습니까?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        # (선택) 이 대화상자에만 별도 스타일을 주고 싶다면 ID 부여 후 CSS에서 #ConfirmExit로 타깃팅
        msg_box.setObjectName("ConfirmExit")

        ## [BEFORE] 인라인 스타일로 색/버튼 모양 지정
        # msg_box.setStyleSheet("""
        #     QMessageBox { background-color: #ffffff; }
        #     QLabel { color: #000000; background-color: #ffffff; }
        #     QPushButton { min-width: 80px; padding: 5px; background-color: #CCCCCC; }
        # """)

        # [NOW] 전역 style.css의 QMessageBox 규칙이 적용되므로 인라인 스타일 불필요

        # 3) 메시지 박스를 실행하고 사용자 응답 받기
        reply = msg_box.exec()

        # 4) 사용자 응답에 따라 처리 (로직은 기존과 동일)
        if reply == QMessageBox.StandardButton.Yes:
            print("애플리케이션을 종료합니다. 모든 스레드와 타이머를 정지합니다.")

            # DB 스레드 종료
            if hasattr(self, "_db_thread") and self._db_thread.isRunning():
                self._db_thread.stop()
                self._db_thread.wait()

            # 모든 타이머 정지
            timers = ["_clock_timer", "_image_timer", "_blink_timer", "_network_timer", "heartbeat_timer",
                      "speed_timer"]
            for timer_name in timers:
                if hasattr(self, timer_name):
                    timer = getattr(self, timer_name)
                    if timer and timer.isActive():
                        timer.stop()

            event.accept()
            super().closeEvent(event)
        else:
            event.ignore()


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------

# def main():
#     # 고DPI 픽스맵 사용 → 작은 아이콘 선명도 개선
#     if QApplication.instance() is None:
#         QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
#     app = QApplication.instance() or QApplication(sys.argv)
#     apply_global_stylesheet(app)
#
#     here = Path(__file__).resolve().parent  # ui/
#     enc_path = here / "path.enc"  # ui/path.enc
#
#     print(f"SYSTEM_PATH={os.environ.get('SYSTEM_PATH')}")
#     print(f"enc_path={enc_path} exists={enc_path.is_file()}")
#
#     try:
#         cfg = AppConfig.load(enc_path=str(enc_path))
#     except Exception as e:
#         m = QMessageBox(QMessageBox.Icon.Critical,
#                         "설정 로드 오류",
#                         f"{type(e).__name__}: {e}",
#                         QMessageBox.StandardButton.Ok)
#         m.setDetailedText(traceback.format_exc())
#         m.exec()
#         return 1
#
#     tz = resolve_kst()
#     win = MainWindow(cfg, tz)
#     win.show()
#     return app.exec()

def main():
    if QApplication.instance() is None:
        QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)

    # 내 파일 바로 옆의 style.css를 우선 지정해 강제 적용
    here = Path(__file__).resolve().parent
    os.environ.setdefault("APP_STYLE_PATH", str(here / "style.css"))

    apply_global_stylesheet(app)

    enc_path = here / "path.enc"
    print(f"SYSTEM_PATH={os.environ.get('SYSTEM_PATH')}")
    print(f"enc_path={enc_path} exists={enc_path.is_file()}")

    try:
        cfg = AppConfig.load(enc_path=str(enc_path))
    except Exception as e:
        m = QMessageBox(QMessageBox.Icon.Critical, "설정 로드 오류",
                        f"{type(e).__name__}: {e}", QMessageBox.StandardButton.Ok)
        m.setDetailedText(traceback.format_exc())
        m.exec()
        return 1

    tz = resolve_kst()
    win = MainWindow(cfg, tz)
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

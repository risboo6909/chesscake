import chess
import chess.engine


class AnalysisResult(object):
    def __init__(self, info: chess.engine.InfoDict):
        self.info = info

    def analysis_ready(self):
        return self.info is not None

    def get_moves(self, max_moves=5):
        if self.info is None or 'pv' not in self.info:
            return None
        return ' '.join([str(move) for move in self.info['pv']][:max_moves*2])

def analyze_board(board: chess.Board):
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/Cellar/stockfish/15/bin/stockfish")
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=20))
    except Exception as e:
        print(e)
        return None
    else:
        engine.quit()
        return info

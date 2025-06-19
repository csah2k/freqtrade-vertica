
import os
import re
import logging
import vertica_python
from datetime import datetime, timedelta

from verticapy import vDataFrame
from verticapy import set_option
from verticapy.sql import insert_into
from verticapy.connection import set_connection
from pandas import DataFrame, to_datetime, read_parquet

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class VerticaDataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def parse_timeframe_to_timedelta(self, timeframe: str) -> timedelta:
        match = re.match(r"(\d+)([mhd])", timeframe)
        if not match:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
        value, unit = int(match.group(1)), match.group(2)
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    def __init__(self, datadir):
        self.required_columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'some_var',
            'symbol', 'base_asset', 'quote_asset'
        ]
        self.db_schema = os.getenv('VERT_SCHEMA', 'stocks')
        self.hist_table = os.getenv('VERT_TABLE_HIST', 'historical_data')
        self.hist_relation = f'"{self.db_schema}"."{self.hist_table}"'
        self.pred_table = os.getenv('VERT_TABLE_PRED', 'trades_predict')
        self.pred_relation = f'"{self.db_schema}"."{self.pred_table}"'

        self.conn_info = {
            'host': os.getenv('VERT_HOST', 'localhost'),
            'port': os.getenv('VERT_PORT', 5433),
            'user': os.getenv('VERT_USER', 'dbadmin'),
            'password': os.getenv('VERT_PASS', 'password'),
            'database': os.getenv('VERT_DB', 'VMart'),
            'tlsmode': os.getenv('VERT_TLS', 'disable'),
            'use_prepared_statements': False,
            'autocommit': True
        }
        self.vertica_connection = vertica_python.connect(**self.conn_info) 
        set_connection(self.vertica_connection)
        set_option("sql_on", False)
        set_option("print_info", False)  

    
    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Store data in Vertica database.
        [[<date>,<open>,<high>,<low>,<close>,<volume>]]
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :param candle_type: !! Not implemented !!
        :return: None
        """
        logging.info(f"vertica_ohlcv_store: {pair}, timeframe: {timeframe}, data: {data.head()}")
        raise NotImplementedError()

        base, quote = pair.split('/')
        symbol = pair.replace("/", "")

        # If input has just 5 columns (date, open, high, low, close, volume), rename and extend
        if data.shape[1] != 6:
            raise Exception(f"Dataframe columns with unsupported size: {data.head()}")
        
        data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        data['quote_asset_volume'] = None
        data['number_of_trades'] = None
        data['taker_buy_base_asset_volume'] = None
        data['taker_buy_quote_asset_volume'] = None
        data['some_var'] = None

        # Ensure open_time is datetime
        #data['open_time'] = to_datetime(data['open_time'])

        # Compute close_time
        delta = self.parse_timeframe_to_timedelta(timeframe)
        data['close_time'] = data['open_time'] + delta

        # Add symbol/base/quote info
        data['symbol'] = symbol
        data['base_asset'] = base
        data['quote_asset'] = quote

        # Ensure all columns are in the right order
        data = data[self.required_columns]

        # Insert into Vertica DB
        rows = insert_into(
            column_names = self.required_columns,
            table_name = self.hist_table,
            schema = self.db_schema,
            data = data,
            genSQL = True # --------- DEBUG ----------
        )
        logging.info(f"rows inserted: {rows}")


    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType
    ) -> DataFrame:
        """
        Internal method used to load data for one pair from Vertica DB.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :param candle_type: !! Not implemented !!
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        try:
            logging.info(f"vertica_ohlcv_load: {pair}, timeframe: {timeframe}, timerange: {timerange.timerange_str if timerange else None}")
            base, quote = pair.split('/')
            startdt = timerange.startdt if timerange and timerange.startdt else datetime.now() - timedelta(days=30)
            stopdt = timerange.stopdt if timerange and timerange.stopdt else datetime.now()
            vdf:vDataFrame = vDataFrame(self.hist_table, schema=self.db_schema)\
                .filter([f"base_asset = '{base}'",
                        f"quote_asset = '{quote}'",
                        f"close_time >= '{startdt.isoformat()}'",
                        f"close_time <= '{stopdt.isoformat()}'"])\
                .interpolate(
                    ts = "close_time",
                    rule = timeframe,
                    method = { 
                        "low": "linear", 
                        "high": "linear",
                        "open": "linear",
                        "close": "linear", 
                        "volume": "linear"
                    })\
                .select(["close_time as ts", "open", "high", "low", "close", "volume"])
           
            vdf.eval("date", "ts::TIMESTAMPTZ")
            df = vdf.select(self._columns).to_pandas()
            return df

        except Exception as e:
            logger.exception(
                f"Error loading data from {pair}. Exception: {e}. Returning empty dataframe."
            )
            return DataFrame(columns=self._columns)

    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        #raise NotImplementedError()
        logging.info(f"vertica_ohlcv_append: {data.head()}")
        self.ohlcv_store(pair, timeframe, data)

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """
        logging.info(f"vertica_trades_store: {pair} : {trading_mode} : {data.head()}")
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).to_parquet(filename)

    def trades_append(self, pair: str, data: DataFrame):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: respect timerange ...
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - currently not implemented
        :return: List of trades
        """
        logging.info(f"vertica_trades_load: {pair} : {trading_mode} : {timerange}")
        # ["timestamp", "id", "type", "side", "price", "amount", "cost"]

        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        tradesdata = read_parquet(filename)

        return tradesdata

    @classmethod
    def _get_file_extension(cls):
        return "vertica"

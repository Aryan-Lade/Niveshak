"""
NSE F&O stock universe used by the Niveshak market terminal.

Each row is ``SYMBOL|Company name|Sector|reference price``. The reference
price is only a starting level for the simulated feed (see ``market_engine``);
it is not a quote. Swap this table for the exchange's current F&O list when a
real data feed is connected - symbols enter and leave the segment every few
months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    sector: str
    ref_price: float


_RAW = """
HDFCBANK|HDFC Bank|Private Banks|985
ICICIBANK|ICICI Bank|Private Banks|1380
AXISBANK|Axis Bank|Private Banks|1210
KOTAKBANK|Kotak Mahindra Bank|Private Banks|2120
INDUSINDBK|IndusInd Bank|Private Banks|790
FEDERALBNK|Federal Bank|Private Banks|232
IDFCFIRSTB|IDFC First Bank|Private Banks|81
BANDHANBNK|Bandhan Bank|Private Banks|168
AUBANK|AU Small Finance Bank|Private Banks|880
RBLBANK|RBL Bank|Private Banks|305
YESBANK|Yes Bank|Private Banks|22.5
SBIN|State Bank of India|PSU Banks|955
BANKBARODA|Bank of Baroda|PSU Banks|285
PNB|Punjab National Bank|PSU Banks|122
CANBK|Canara Bank|PSU Banks|138
UNIONBANK|Union Bank of India|PSU Banks|152
INDIANB|Indian Bank|PSU Banks|820
BANKINDIA|Bank of India|PSU Banks|142
BAJFINANCE|Bajaj Finance|NBFC|1040
BAJAJFINSV|Bajaj Finserv|NBFC|2080
SHRIRAMFIN|Shriram Finance|NBFC|780
CHOLAFIN|Cholamandalam Investment|NBFC|1690
MUTHOOTFIN|Muthoot Finance|NBFC|3250
M&MFIN|Mahindra & Mahindra Financial|NBFC|310
LICHSGFIN|LIC Housing Finance|NBFC|575
PFC|Power Finance Corporation|NBFC|395
RECLTD|REC|NBFC|365
HUDCO|Housing & Urban Development Corp|NBFC|232
IRFC|Indian Railway Finance Corp|NBFC|124
ABCAPITAL|Aditya Birla Capital|NBFC|325
LTF|L&T Finance|NBFC|285
MANAPPURAM|Manappuram Finance|NBFC|285
PNBHOUSING|PNB Housing Finance|NBFC|905
JIOFIN|Jio Financial Services|NBFC|305
SBICARD|SBI Cards|NBFC|880
POONAWALLA|Poonawalla Fincorp|NBFC|480
LICI|Life Insurance Corporation|Insurance|905
SBILIFE|SBI Life Insurance|Insurance|1980
HDFCLIFE|HDFC Life Insurance|Insurance|760
ICICIPRULI|ICICI Prudential Life|Insurance|620
ICICIGI|ICICI Lombard General Insurance|Insurance|1990
MFSL|Max Financial Services|Insurance|1620
BSE|BSE|Capital Markets|2650
CDSL|Central Depository Services|Capital Markets|1620
MCX|Multi Commodity Exchange|Capital Markets|9400
ANGELONE|Angel One|Capital Markets|2650
HDFCAMC|HDFC Asset Management|Capital Markets|5450
CAMS|Computer Age Management Services|Capital Markets|3900
KFINTECH|KFin Technologies|Capital Markets|1100
NUVAMA|Nuvama Wealth Management|Capital Markets|7200
360ONE|360 One WAM|Capital Markets|1130
IEX|Indian Energy Exchange|Capital Markets|142
TCS|Tata Consultancy Services|IT|3080
INFY|Infosys|IT|1510
HCLTECH|HCL Technologies|IT|1590
WIPRO|Wipro|IT|245
TECHM|Tech Mahindra|IT|1440
LTIM|LTIMindtree|IT|5700
PERSISTENT|Persistent Systems|IT|5900
COFORGE|Coforge|IT|1780
MPHASIS|Mphasis|IT|2800
OFSS|Oracle Financial Services|IT|8600
KPITTECH|KPIT Technologies|IT|1180
TATAELXSI|Tata Elxsi|IT|5400
CYIENT|Cyient|IT|1160
TATATECH|Tata Technologies|IT|690
MARUTI|Maruti Suzuki|Auto|15800
M&M|Mahindra & Mahindra|Auto|3650
TMPV|Tata Motors Passenger Vehicles|Auto|410
BAJAJ-AUTO|Bajaj Auto|Auto|8900
HEROMOTOCO|Hero MotoCorp|Auto|5500
EICHERMOT|Eicher Motors|Auto|6900
TVSMOTOR|TVS Motor|Auto|3500
ASHOKLEY|Ashok Leyland|Auto|140
BOSCHLTD|Bosch|Auto|37500
MOTHERSON|Samvardhana Motherson|Auto|105
BHARATFORG|Bharat Forge|Auto|1320
EXIDEIND|Exide Industries|Auto|385
TIINDIA|Tube Investments of India|Auto|3100
SONACOMS|Sona BLW Precision Forgings|Auto|480
UNOMINDA|Uno Minda|Auto|1280
SUNPHARMA|Sun Pharmaceutical|Pharma & Health|1700
DRREDDY|Dr. Reddy's Laboratories|Pharma & Health|1250
CIPLA|Cipla|Pharma & Health|1520
DIVISLAB|Divi's Laboratories|Pharma & Health|6400
LUPIN|Lupin|Pharma & Health|1980
AUROPHARMA|Aurobindo Pharma|Pharma & Health|1130
ZYDUSLIFE|Zydus Lifesciences|Pharma & Health|960
TORNTPHARM|Torrent Pharmaceuticals|Pharma & Health|3600
ALKEM|Alkem Laboratories|Pharma & Health|5600
BIOCON|Biocon|Pharma & Health|390
GLENMARK|Glenmark Pharmaceuticals|Pharma & Health|1850
LAURUSLABS|Laurus Labs|Pharma & Health|920
MANKIND|Mankind Pharma|Pharma & Health|2400
APOLLOHOSP|Apollo Hospitals|Pharma & Health|7700
MAXHEALTH|Max Healthcare|Pharma & Health|1150
FORTIS|Fortis Healthcare|Pharma & Health|1020
SYNGENE|Syngene International|Pharma & Health|640
HINDUNILVR|Hindustan Unilever|FMCG|2450
ITC|ITC|FMCG|410
NESTLEIND|Nestle India|FMCG|1270
BRITANNIA|Britannia Industries|FMCG|5900
DABUR|Dabur India|FMCG|510
MARICO|Marico|FMCG|720
GODREJCP|Godrej Consumer Products|FMCG|1150
COLPAL|Colgate-Palmolive India|FMCG|2250
TATACONSUM|Tata Consumer Products|FMCG|1160
VBL|Varun Beverages|FMCG|470
UNITDSPR|United Spirits|FMCG|1380
PATANJALI|Patanjali Foods|FMCG|590
TITAN|Titan Company|Consumer|3750
ASIANPAINT|Asian Paints|Consumer|2500
DMART|Avenue Supermarts|Consumer|4200
TRENT|Trent|Consumer|4700
PIDILITIND|Pidilite Industries|Consumer|1480
HAVELLS|Havells India|Consumer|1480
VOLTAS|Voltas|Consumer|1350
DIXON|Dixon Technologies|Consumer|15500
PAGEIND|Page Industries|Consumer|41000
JUBLFOOD|Jubilant FoodWorks|Consumer|610
KALYANKJIL|Kalyan Jewellers|Consumer|510
BLUESTARCO|Blue Star|Consumer|1900
CROMPTON|Crompton Greaves Consumer|Consumer|290
INDHOTEL|Indian Hotels|Consumer|740
SUPREMEIND|Supreme Industries|Consumer|4000
TATASTEEL|Tata Steel|Metals & Mining|178
JSWSTEEL|JSW Steel|Metals & Mining|1160
HINDALCO|Hindalco Industries|Metals & Mining|790
VEDL|Vedanta|Metals & Mining|505
JINDALSTEL|Jindal Steel|Metals & Mining|1050
SAIL|Steel Authority of India|Metals & Mining|135
NMDC|NMDC|Metals & Mining|76
NATIONALUM|National Aluminium|Metals & Mining|235
HINDZINC|Hindustan Zinc|Metals & Mining|480
APLAPOLLO|APL Apollo Tubes|Metals & Mining|1750
COALINDIA|Coal India|Metals & Mining|385
RELIANCE|Reliance Industries|Oil & Gas|1490
ONGC|Oil & Natural Gas Corporation|Oil & Gas|250
IOC|Indian Oil Corporation|Oil & Gas|165
BPCL|Bharat Petroleum|Oil & Gas|365
HINDPETRO|Hindustan Petroleum|Oil & Gas|465
GAIL|GAIL India|Oil & Gas|180
PETRONET|Petronet LNG|Oil & Gas|280
OIL|Oil India|Oil & Gas|430
IGL|Indraprastha Gas|Oil & Gas|210
NTPC|NTPC|Power|330
POWERGRID|Power Grid Corporation|Power|285
TATAPOWER|Tata Power|Power|395
ADANIGREEN|Adani Green Energy|Power|1040
ADANIENSOL|Adani Energy Solutions|Power|950
JSWENERGY|JSW Energy|Power|520
NHPC|NHPC|Power|82
SJVN|SJVN|Power|88
TORNTPOWER|Torrent Power|Power|1300
SUZLON|Suzlon Energy|Power|57
INOXWIND|Inox Wind|Power|150
IREDA|Indian Renewable Energy Dev Agency|Power|150
LT|Larsen & Toubro|Capital Goods|3950
SIEMENS|Siemens|Capital Goods|3150
ABB|ABB India|Capital Goods|5150
CGPOWER|CG Power & Industrial Solutions|Capital Goods|740
BHEL|Bharat Heavy Electricals|Capital Goods|265
CUMMINSIND|Cummins India|Capital Goods|4300
POWERINDIA|Hitachi Energy India|Capital Goods|19500
KAYNES|Kaynes Technology|Capital Goods|6500
POLYCAB|Polycab India|Capital Goods|7600
KEI|KEI Industries|Capital Goods|4100
ASTRAL|Astral|Capital Goods|1450
HAL|Hindustan Aeronautics|Defence|4750
BEL|Bharat Electronics|Defence|420
BDL|Bharat Dynamics|Defence|1520
MAZDOCK|Mazagon Dock Shipbuilders|Defence|2750
SOLARINDS|Solar Industries India|Defence|13800
ULTRACEMCO|UltraTech Cement|Cement|11900
GRASIM|Grasim Industries|Cement|2850
SHREECEM|Shree Cement|Cement|28500
AMBUJACEM|Ambuja Cements|Cement|560
DALBHARAT|Dalmia Bharat|Cement|2100
SRF|SRF|Chemicals|2950
PIIND|PI Industries|Chemicals|3600
UPL|UPL|Chemicals|740
TATACHEM|Tata Chemicals|Chemicals|880
DLF|DLF|Realty|760
GODREJPROP|Godrej Properties|Realty|2250
LODHA|Lodha Developers|Realty|1200
OBEROIRLTY|Oberoi Realty|Realty|1700
PRESTIGE|Prestige Estates|Realty|1750
PHOENIXLTD|Phoenix Mills|Realty|1700
BHARTIARTL|Bharti Airtel|Telecom|2050
IDEA|Vodafone Idea|Telecom|9.5
INDUSTOWER|Indus Towers|Telecom|395
TATACOMM|Tata Communications|Telecom|1850
ETERNAL|Eternal|New-age Tech|310
PAYTM|One 97 Communications|New-age Tech|1280
NYKAA|FSN E-Commerce Ventures|New-age Tech|250
POLICYBZR|PB Fintech|New-age Tech|1800
NAUKRI|Info Edge India|New-age Tech|1350
SWIGGY|Swiggy|New-age Tech|410
ADANIPORTS|Adani Ports & SEZ|Infra & Logistics|1470
ADANIENT|Adani Enterprises|Infra & Logistics|2450
CONCOR|Container Corporation of India|Infra & Logistics|530
GMRAIRPORT|GMR Airports|Infra & Logistics|95
INDIGO|InterGlobe Aviation|Infra & Logistics|5800
IRCTC|Indian Railway Catering & Tourism|Infra & Logistics|720
RVNL|Rail Vikas Nigam|Infra & Logistics|320
NBCC|NBCC India|Infra & Logistics|115
DELHIVERY|Delhivery|Infra & Logistics|470
"""


def _parse(raw: str) -> List[Instrument]:
    rows: List[Instrument] = []
    seen = set()
    for line in raw.strip().splitlines():
        symbol, name, sector, price = (part.strip() for part in line.split("|"))
        if symbol in seen:
            raise ValueError(f"Duplicate symbol in universe: {symbol}")
        seen.add(symbol)
        rows.append(Instrument(symbol=symbol, name=name, sector=sector, ref_price=float(price)))
    return rows


UNIVERSE: List[Instrument] = _parse(_RAW)
BY_SYMBOL: Dict[str, Instrument] = {i.symbol: i for i in UNIVERSE}
SECTORS: List[str] = sorted({i.sector for i in UNIVERSE})

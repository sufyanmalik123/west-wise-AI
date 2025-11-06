from google.ads.googleads.client import GoogleAdsClient
from datetime import datetime, timedelta
import dlt
import os
from google.protobuf.json_format import MessageToDict
from pytz import timezone
from dotenv import load_dotenv
import json
load_dotenv()
from .queries import (
    calls_query , ad_group_ads_query , ad_group_ads_report_query ,
    keywords_reports_query , campaigns_query
)
config = {
    "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
    "client_id": os.getenv("GOOGLE_ADS_CLIENT_ID"),
    "client_secret": os.getenv("GOOGLE_ADS_CLIENT_SECRET"),
    "refresh_token": os.getenv("GOOGLE_ADS_REFRESH_TOKEN"),
    "login_customer_id": os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
    "use_proto_plus": True,
}
google_ads_client = GoogleAdsClient.load_from_dict(config)
customer_id = "2762589697"
all_ids = ["4738040605" , "9409120195" , customer_id]
# Incremental load
# start_date = datetime.today()-timedelta(days=7)
# end_date = datetime.today()
# #converting the timezone from PKT to PST
# start_date = start_date.astimezone(timezone('US/Pacific'))
# end_date = end_date.astimezone(timezone('US/Pacific'))
# start_date_format = start_date.strftime('%Y-%m-%d')
# end_date_format = end_date.strftime('%Y-%m-%d')
# start_date_time_format = start_date.strftime('%Y-%m-%d 00:00:00')
# end_date_time_format = end_date.strftime('%Y-%m-%d 23:59:59')
# Historical load
start_date_time_format = "2025-07-01 00:00:00"
end_date_time_format = "2025-12-24 23:59:59"
start_date_format = '2025-07-01'
end_date_format = '2025-12-24'
def fetch_google_ads_data(query , customer_id):
    ga_service = google_ads_client.get_service("GoogleAdsService")
    print("GoogleAds: querying...")
    response = ga_service.search_stream(customer_id=customer_id, query=query)
    print("GoogleAds: response", response)
    data = []
    for batch in response:
        for row in batch.results:
            temp = MessageToDict(row._pb)
            for key, value in temp.items():
                # if the value against a key in the dictionary is a list, preventing it's separate table creation
                if isinstance(value, list):
                    temp[key] = json.dumps(value)
            data.append(temp)
    print("Records fetched", len(data))
    return data
#@dlt.resource(name="campaigns", max_table_nesting=1, primary_key=["campaign__resource_name" , "campaign__campaign_budget" , "campaign_budget__resource_name" , "campaign__start_date" , "campaign__end_date" , "segments__date" , "segments__hour"] ,write_disposition={"disposition": "merge", "strategy": "delete-insert", "deduplicated": True})
@dlt.resource(name="campaigns", max_table_nesting=1, write_disposition="replace")
def campaigns():
    yield fetch_google_ads_data(campaigns_query(start_date_format , end_date_format) , customer_id)
#@dlt.resource(name="calls" , max_table_nesting=1 , primary_key="call_view__resource_name" , write_disposition={"disposition": "merge", "strategy": "delete-insert", "deduplicated": True})
@dlt.resource(name="calls" , max_table_nesting=1, write_disposition="replace")
def call_views():
    yield fetch_google_ads_data(calls_query(start_date_time_format , end_date_time_format) , customer_id)
#@dlt.resource(name="ad_group_ads" , max_table_nesting=1 , primary_key=["ad_group_ad__resource_name" , "segments__date" , "ad_group_ad__ad_group"] , write_disposition={"disposition": "merge", "strategy": "delete-insert", "deduplicated": True})
@dlt.resource(name="ad_group_ads" , max_table_nesting=1 , write_disposition="replace")
def ad_group_ads():
    yield fetch_google_ads_data(ad_group_ads_query(start_date_format , end_date_format) , customer_id)
#@dlt.resource(name="ad_group_ad_report" , max_table_nesting=1 , primary_key=["campaign__resource_name", "ad_group_ad__resource_name", "segments__date"] , write_disposition={"disposition": "merge", "strategy": "delete-insert", "deduplicated": True})
@dlt.resource(name="ad_group_ad_report" , max_table_nesting=1 , write_disposition="replace")
def ad_group_ad_report():
    for id in all_ids:
        yield fetch_google_ads_data(ad_group_ads_report_query(start_date_format , end_date_format) , id)
#@dlt.resource(name="keyword_report" , max_table_nesting=1 ,  primary_key=["campaign__resource_name", "ad_group__resource_name" , "ad_group_criterion__resource_name" ,"keyword_view__resource_name" , "segments__date"] , write_disposition={"disposition": "merge", "strategy": "delete-insert", "deduplicated": True})
@dlt.resource(name="keyword_report" , max_table_nesting=1 ,  write_disposition="replace")
def keyword_report_source():
    for id in all_ids:
        yield fetch_google_ads_data(keywords_reports_query(start_date_format , end_date_format) , id)
@dlt.source
def source_google_ads_raw():
    #return campaigns ,ad_group_ads , ad_group_ad_report, keyword_report_source , call_views
    return ad_group_ad_report , keyword_report_source
def pipeline_google_ads():
    pipeline = dlt.pipeline(
        pipeline_name="google_ads_raw_pipeline",
        destination="bigquery",
        dataset_name="google_ads_raw_test",
        progress="log"
    )
    loadInfo = pipeline.run(source_google_ads_raw())
    print('GoogleAds: loadInfo', loadInfo)
    # fetch_google_ads_data()
pipeline_google_ads()






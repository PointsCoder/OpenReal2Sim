import requests

ntfy_url = ""

def notify_success(pipeline_label):
    try:
        requests.post(
            ntfy_url, 
            data=pipeline_label,
            headers={
                "Title": f"Reconstruction completed",
                "Tags": "white_check_mark"      
            }
        )
    except:
        pass

def notify_started(pipeline_label):
    try:
        requests.post(
            ntfy_url, 
            data=pipeline_label,
            headers={
                "Title": f"Reconstruction started",
                "Tags": "watch"      
            }
        )
    except:
        pass

def notify_failed(pipeline_label):
    try:
        requests.post(
            ntfy_url, 
            data=pipeline_label,
            headers={
                "Title": f"Reconstruction failed",
                "Tags": "x"      
            }
        )
    except:
        pass
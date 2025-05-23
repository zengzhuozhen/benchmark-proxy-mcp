from fastmcp import FastMCP
from typing import Optional, Dict, Union, List
import requests
import subprocess
import re


mcp = FastMCP("benchmark-proxy mcp")

def format_status_checker(status: Union[int, str, List[int]]) -> str:
    """
    Format status checker expression based on input type.
    Examples:
    - Single value: 200
    - Multiple values: 200,201
    - Range: 200-299
    """
    if isinstance(status, int):
        return str(status)
    elif isinstance(status, list):
        return ",".join(map(str, status))
    return str(status)

def format_body_checker(body_check: str) -> str:
    """
    Format body checker expression.
    Examples:
    - Exact match: hello world
    - Contains: @Contains[success]
    - Regex: @Reg[\w+]
    """
    if body_check.startswith("@Contains["):
        return body_check
    elif body_check.startswith("@Reg["):
        return body_check
    return body_check

@mcp.tool()
def call(backendAPI: str,headers: dict) -> str:
    """
    Call a benchmark test job for the backendAPI using benchmark-proxy service.
    Example: curl https://www.baidu.com \
        -H 'Benchmark-Proxy-Times:1' \
        -H 'Benchmark-Proxy-Concurrency:1'

    Args:
        backendAPI: The target API to test.
        headers: Custom benchmark headers such as:
            - Benchmark-Proxy-Times
            - Benchmark-Proxy-Duration
            - Benchmark-Proxy-Concurrency
            - Benchmark-Proxy-Check-Result-Status
            - Benchmark-Proxy-Check-Result-Body

    Returns:
        A string response returned by benchmark-proxy (may include result summary).
    """
    proxy_host = "http://127.0.0.1:9900"
    result_text = ""

    # return result_text
    try:
        # 构建 curl 命令
        curl_cmd = ["curl", "-s", "-i", "-m", "60", "-A", "FastMCP-Client/1.0"]
        curl_cmd.extend(["-x", proxy_host])  # 设置代理
        curl_cmd.append(backendAPI)  # 目标 URL

        # 添加自定义 headers
        for key, value in headers.items():
            curl_cmd.extend(["-H", f"{key}: {value}"])

        # 执行 curl 命令
        result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # Check if curl command resulted in an error indicated by non-zero exit code
        if result.returncode != 0:
            # Attempt to extract error message from stderr if available
            error_message = result.stderr.strip() if result.stderr else "No stderr output."
            # Also include stdout if it contains relevant info (like proxy error messages)
            stdout_info = result.stdout.strip() if result.stdout else "No stdout output."
            return f"Error: curl command failed with exit code {result.returncode}. Stderr: '{error_message}'. Stdout: '{stdout_info}'"

        # 解析 curl 输出 (stdout contains headers and body)
        return result.stdout

    except subprocess.TimeoutExpired:
        return f"Error: curl command timed out after 60 seconds for {backendAPI}"
    except FileNotFoundError:
        return "Error: 'curl' command not found. Please ensure curl is installed and in your PATH."
    except Exception as e:
        # 捕获其他异常
        result_text = f"Error executing benchmark call: {str(e)}"
        return result_text

@mcp.tool()
def run_duration_test(
    url: str,
    duration_seconds: int,
    concurrency: int,
    expected_status: Optional[Union[int, str, List[int]]] = None,
    expected_body: Optional[str] = None
) -> str:
    """
    Runs a benchmark test for a specified duration and concurrency.

    Args:
        url: The target URL to test.
        duration_seconds: How long the test should run in seconds.
        concurrency: The number of concurrent requests.
        expected_status: Optional HTTP status code(s) expected for success. Can be:
            - Single value (e.g., 200)
            - Multiple values (e.g., [200, 201])
            - Range string (e.g., "200-299")
        expected_body: Optional response body check. Can be:
            - Exact match (e.g., "hello world")
            - Contains check (e.g., "@Contains[success]")
            - Regex pattern (e.g., "@Reg[\w+]")

    Returns:
        The raw output string from the benchmark-proxy service.

    Note:
        You can use customize tags in your request data to generate dynamic values:
        - ${uuid} - Random UUID
        - ${int} or ${int:min,max} - Random integer
        - ${float} or ${float:min,max} - Random float
        - ${string} or ${string:length} - Random string
        - ${bool} - Random boolean
        - ${date:format} - Current date with format
        - ${timestamp} - Current unix timestamp
        - ${incr} or ${incr:start,step} - Auto-increment integer
        - ${range:start,end} - Range auto-increment
        - ${list:[a,b,c]} - Random pick from list
        - ${const:value} - Constant value

        Example request data with tags:
        {
            "user_id": "${uuid}",
            "age": "${int:18,60}",
            "name": "${string:8}",
            "email": "${string:6}@example.com",
            "status": "${list:[active,inactive]}",
            "created_at": "${timestamp}"
        }
    """
    benchmark_headers = {
        "Benchmark-Proxy-Duration": str(duration_seconds),
        "Benchmark-Proxy-Concurrency": str(concurrency)
    }
    
    if expected_status is not None:
        benchmark_headers["Benchmark-Proxy-Check-Result-Status"] = format_status_checker(expected_status)
    if expected_body is not None:
        benchmark_headers["Benchmark-Proxy-Check-Result-Body"] = format_body_checker(expected_body)

    return call(backendAPI=url, headers=benchmark_headers)

@mcp.tool()
def run_times_test(
    url: str,
    times: int,
    concurrency: int,
    expected_status: Optional[Union[int, str, List[int]]] = None,
    expected_body: Optional[str] = None
) -> str:
    """
    Runs a benchmark test for a specified number of times and concurrency.

    Args:
        url: The target URL to test.
        times: How many times the request should be executed in total.
        concurrency: The number of concurrent requests.
        expected_status: Optional HTTP status code(s) expected for success. Can be:
            - Single value (e.g., 200)
            - Multiple values (e.g., [200, 201])
            - Range string (e.g., "200-299")
        expected_body: Optional response body check. Can be:
            - Exact match (e.g., "hello world")
            - Contains check (e.g., "@Contains[success]")
            - Regex pattern (e.g., "@Reg[\w+]")

    Returns:
        The raw output string from the benchmark-proxy service.

    Note:
        You can use customize tags in your request data to generate dynamic values:
        - ${uuid} - Random UUID
        - ${int} or ${int:min,max} - Random integer
        - ${float} or ${float:min,max} - Random float
        - ${string} or ${string:length} - Random string
        - ${bool} - Random boolean
        - ${date:format} - Current date with format
        - ${timestamp} - Current unix timestamp
        - ${incr} or ${incr:start,step} - Auto-increment integer
        - ${range:start,end} - Range auto-increment
        - ${list:[a,b,c]} - Random pick from list
        - ${const:value} - Constant value

        Example request data with tags:
        {
            "user_id": "${uuid}",
            "age": "${int:18,60}",
            "name": "${string:8}",
            "email": "${string:6}@example.com",
            "status": "${list:[active,inactive]}",
            "created_at": "${timestamp}"
        }
    """
    benchmark_headers = {
        "Benchmark-Proxy-Times": str(times),
        "Benchmark-Proxy-Concurrency": str(concurrency)
    }
    
    if expected_status is not None:
        benchmark_headers["Benchmark-Proxy-Check-Result-Status"] = format_status_checker(expected_status)
    if expected_body is not None:
        benchmark_headers["Benchmark-Proxy-Check-Result-Body"] = format_body_checker(expected_body)

    return call(backendAPI=url, headers=benchmark_headers)

if __name__ == "__main__":
    mcp.run()

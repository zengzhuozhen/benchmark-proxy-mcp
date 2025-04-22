from fastmcp import FastMCP
from typing import Optional, Dict
import requests
import subprocess


mcp = FastMCP("benchmark-proxy mcp")

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
    expected_status: Optional[int] = None,
    expected_body_contains: Optional[str] = None
) -> str:
    """
    Runs a benchmark test for a specified duration and concurrency.

    Args:
        url: The target URL to test.
        duration_seconds: How long the test should run in seconds.
        concurrency: The number of concurrent requests.
        expected_status: Optional HTTP status code expected for a successful request.
        expected_body_contains: Optional string that the response body must contain for success.

    Returns:
        The raw output string from the benchmark-proxy service.
    """
    benchmark_headers = {
        "Benchmark-Proxy-Duration": str(duration_seconds),
        "Benchmark-Proxy-Concurrency": str(concurrency)
    }
    if expected_status is not None:
        benchmark_headers["Benchmark-Proxy-Check-Result-Status"] = str(expected_status)
    if expected_body_contains is not None:
        # For simplicity, we'll assume direct string match. Use '@Reg:' prefix for regex.
        benchmark_headers["Benchmark-Proxy-Check-Result-Body"] = expected_body_contains

    return call(backendAPI=url, headers=benchmark_headers)

@mcp.tool()
def run_times_test(
    url: str,
    times: int,
    concurrency: int,
    expected_status: Optional[int] = None,
    expected_body_contains: Optional[str] = None
) -> str:
    """
    Runs a benchmark test for a specified number of times and concurrency.

    Args:
        url: The target URL to test.
        times: How many times the request should be executed in total.
        concurrency: The number of concurrent requests.
        expected_status: Optional HTTP status code expected for a successful request.
        expected_body_contains: Optional string that the response body must contain for success.

    Returns:
        The raw output string from the benchmark-proxy service.
    """
    benchmark_headers = {
        "Benchmark-Proxy-Times": str(times),
        "Benchmark-Proxy-Concurrency": str(concurrency)
    }
    if expected_status is not None:
        benchmark_headers["Benchmark-Proxy-Check-Result-Status"] = str(expected_status)
    if expected_body_contains is not None:
        # For simplicity, we'll assume direct string match. Use '@Reg:' prefix for regex.
        benchmark_headers["Benchmark-Proxy-Check-Result-Body"] = expected_body_contains

    return call(backendAPI=url, headers=benchmark_headers)

if __name__ == "__main__":
    mcp.run()

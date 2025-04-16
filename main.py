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
        curl_cmd = ["curl", "-s", "-i", "-m", "30"]  # -s: 静默, -i: 包含响应头, -m 30: 超时 30 秒
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
            check=True
        )

        # 解析 curl 输出
        return result.stdout

    except subprocess.CalledProcessError as e:
        # 捕获 curl 命令执行失败的错误
        result_text = f"Error: curl command failed with exit code {e.returncode}: {e.stderr}"
    except Exception as e:
        # 捕获其他异常
        result_text = f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()

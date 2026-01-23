"""Authentication tools for DerivaML MCP server.

Provides tools for authenticating with Deriva servers using either
Globus Auth or Credenza authentication methods.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("deriva-ml-mcp")


def register_auth_tools(mcp: "FastMCP") -> None:
    """Register authentication tools with the MCP server."""

    @mcp.tool()
    async def globus_login(
        hostname: str,
        refresh: bool = True,
        force: bool = False,
    ) -> str:
        """Login to a Deriva server using Globus Auth.

        Initiates a Globus authentication flow to obtain access tokens for the
        specified host. This will open a browser window for authentication unless
        running in a headless environment.

        **When to use**: Most Deriva deployments use Globus Auth. Use this when
        connecting to servers like dev.eye-ai.org, www.atlas-d2k.org, etc.

        **Important**: This command launches an interactive browser-based login.
        The user must complete the authentication in their browser.

        Args:
            hostname: Server hostname to authenticate with (e.g., "dev.eye-ai.org").
            refresh: If True (default), enable refresh tokens for extended sessions.
            force: If True, force re-authentication even if valid tokens exist.

        Returns:
            JSON with:
            - status: "success", "error", or "pending"
            - hostname: The server authenticated with
            - message: Status message or instructions

        Example:
            globus_login("dev.eye-ai.org")
            -> Opens browser for Globus login, returns success when complete
        """
        try:
            cmd = ["deriva-globus-auth-utils", "login", "--host", hostname]

            if refresh:
                cmd.append("--refresh")
            if force:
                cmd.append("--force")

            # Run the login command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for interactive login
            )

            if result.returncode == 0:
                return json.dumps(
                    {
                        "status": "success",
                        "hostname": hostname,
                        "message": "Successfully authenticated with Globus Auth.",
                    }
                )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Login failed"
                return json.dumps(
                    {
                        "status": "error",
                        "hostname": hostname,
                        "message": error_msg,
                    }
                )

        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": "Login timed out. Please try again and complete the browser authentication promptly.",
                }
            )
        except FileNotFoundError:
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": "deriva-globus-auth-utils not found. Please install deriva-py.",
                }
            )
        except Exception as e:
            logger.error(f"Globus login failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def globus_logout(
        hostname: str | None = None,
    ) -> str:
        """Logout from Globus Auth and revoke tokens.

        Revokes access tokens for the specified host, or all tokens if no host
        is specified.

        Args:
            hostname: Optional server hostname to logout from. If not specified,
                revokes all Globus tokens.

        Returns:
            JSON with status and message.

        Example:
            globus_logout("dev.eye-ai.org")  -> Logout from specific host
            globus_logout()  -> Logout from all hosts
        """
        try:
            cmd = ["deriva-globus-auth-utils", "logout"]

            if hostname:
                cmd.extend(["--host", hostname])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                target = hostname if hostname else "all hosts"
                return json.dumps(
                    {
                        "status": "success",
                        "message": f"Successfully logged out from {target}.",
                    }
                )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Logout failed"
                return json.dumps(
                    {
                        "status": "error",
                        "message": error_msg,
                    }
                )

        except FileNotFoundError:
            return json.dumps(
                {
                    "status": "error",
                    "message": "deriva-globus-auth-utils not found. Please install deriva-py.",
                }
            )
        except Exception as e:
            logger.error(f"Globus logout failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def globus_user_info() -> str:
        """Get information about the currently logged-in Globus user.

        Retrieves user identity information from the current Globus Auth session.

        Returns:
            JSON with user information including:
            - status: "success" or "error"
            - user_info: User identity details (if logged in)
            - message: Error message (if not logged in)

        Example:
            globus_user_info()
            -> {"status": "success", "user_info": {"name": "...", "email": "..."}}
        """
        try:
            result = subprocess.run(
                ["deriva-globus-auth-utils", "user-info", "--pretty"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                try:
                    user_info = json.loads(result.stdout)
                    return json.dumps(
                        {
                            "status": "success",
                            "user_info": user_info,
                        }
                    )
                except json.JSONDecodeError:
                    return json.dumps(
                        {
                            "status": "success",
                            "user_info": result.stdout.strip(),
                        }
                    )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Not logged in"
                return json.dumps(
                    {
                        "status": "error",
                        "message": error_msg,
                    }
                )

        except FileNotFoundError:
            return json.dumps(
                {
                    "status": "error",
                    "message": "deriva-globus-auth-utils not found. Please install deriva-py.",
                }
            )
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def credenza_login(
        hostname: str,
        refresh: bool = True,
        force: bool = False,
    ) -> str:
        """Login to a Deriva server using Credenza Auth (device flow).

        Initiates a Credenza authentication flow using device code authorization.
        This displays a URL and code that the user must enter in their browser.

        **When to use**: Some Deriva deployments use Credenza instead of Globus.
        Use this if globus_login doesn't work for your server.

        **Important**: This command initiates a device flow login. Follow the
        displayed instructions to complete authentication in your browser.

        Args:
            hostname: Server hostname to authenticate with (e.g., "example.org").
            refresh: If True (default), enable refresh tokens for extended sessions.
            force: If True, force re-authentication even if valid tokens exist.

        Returns:
            JSON with:
            - status: "success" or "error"
            - hostname: The server authenticated with
            - message: Status message or instructions

        Example:
            credenza_login("example.org")
            -> Displays device code, returns success when authentication complete
        """
        try:
            cmd = ["deriva-credenza-auth-utils", "--host", hostname, "login"]

            if refresh:
                cmd.append("--refresh")
            if force:
                cmd.append("--force")

            # Run the login command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for interactive login
            )

            if result.returncode == 0:
                return json.dumps(
                    {
                        "status": "success",
                        "hostname": hostname,
                        "message": "Successfully authenticated with Credenza Auth.",
                    }
                )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Login failed"
                return json.dumps(
                    {
                        "status": "error",
                        "hostname": hostname,
                        "message": error_msg,
                    }
                )

        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": "Login timed out. Please try again and complete the device code authentication promptly.",
                }
            )
        except FileNotFoundError:
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": "deriva-credenza-auth-utils not found. Please install deriva-py.",
                }
            )
        except Exception as e:
            logger.error(f"Credenza login failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def credenza_logout(
        hostname: str,
    ) -> str:
        """Logout from Credenza Auth and revoke tokens.

        Revokes all access and refresh tokens for the specified host.

        Args:
            hostname: Server hostname to logout from.

        Returns:
            JSON with status and message.

        Example:
            credenza_logout("example.org")
        """
        try:
            result = subprocess.run(
                ["deriva-credenza-auth-utils", "--host", hostname, "logout"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return json.dumps(
                    {
                        "status": "success",
                        "hostname": hostname,
                        "message": f"Successfully logged out from {hostname}.",
                    }
                )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Logout failed"
                return json.dumps(
                    {
                        "status": "error",
                        "hostname": hostname,
                        "message": error_msg,
                    }
                )

        except FileNotFoundError:
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": "deriva-credenza-auth-utils not found. Please install deriva-py.",
                }
            )
        except Exception as e:
            logger.error(f"Credenza logout failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def credenza_session_info(
        hostname: str,
    ) -> str:
        """Get information about the current Credenza session.

        Retrieves session information for the specified host.

        Args:
            hostname: Server hostname to get session info for.

        Returns:
            JSON with session information including:
            - status: "success" or "error"
            - session_info: Session details (if logged in)
            - message: Error message (if not logged in)

        Example:
            credenza_session_info("example.org")
            -> {"status": "success", "session_info": {...}}
        """
        try:
            result = subprocess.run(
                ["deriva-credenza-auth-utils", "--host", hostname, "get-session", "--pretty"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                try:
                    session_info = json.loads(result.stdout)
                    return json.dumps(
                        {
                            "status": "success",
                            "hostname": hostname,
                            "session_info": session_info,
                        }
                    )
                except json.JSONDecodeError:
                    return json.dumps(
                        {
                            "status": "success",
                            "hostname": hostname,
                            "session_info": result.stdout.strip(),
                        }
                    )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Not logged in or session expired"
                return json.dumps(
                    {
                        "status": "error",
                        "hostname": hostname,
                        "message": error_msg,
                    }
                )

        except FileNotFoundError:
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": "deriva-credenza-auth-utils not found. Please install deriva-py.",
                }
            )
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def check_auth_status(
        hostname: str,
    ) -> str:
        """Check authentication status for a Deriva server.

        Attempts to determine if valid credentials exist for the specified host
        by checking both Globus and Credenza authentication.

        Args:
            hostname: Server hostname to check authentication for.

        Returns:
            JSON with:
            - status: "authenticated", "not_authenticated", or "error"
            - hostname: The server checked
            - auth_method: "globus", "credenza", or None
            - message: Status details

        Example:
            check_auth_status("dev.eye-ai.org")
            -> {"status": "authenticated", "auth_method": "globus", ...}
        """
        try:
            from deriva.core import get_credential

            credential = get_credential(hostname)

            if credential:
                # Try to determine which auth method
                return json.dumps(
                    {
                        "status": "authenticated",
                        "hostname": hostname,
                        "message": "Valid credentials found for this host.",
                    }
                )
            else:
                return json.dumps(
                    {
                        "status": "not_authenticated",
                        "hostname": hostname,
                        "message": "No valid credentials found. Use globus_login or credenza_login to authenticate.",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to check auth status: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "hostname": hostname,
                    "message": str(e),
                }
            )

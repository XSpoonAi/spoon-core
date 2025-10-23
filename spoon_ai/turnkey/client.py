import json
import time
import os
import subprocess
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from base64 import urlsafe_b64encode
import requests


class Turnkey:
    """Turnkey API client class for managing blockchain private keys and wallet operations."""

    def __init__(self, base_url=None, api_public_key=None, api_private_key=None, org_id=None):
        """
        Initialize Turnkey client.

        Args:
            base_url (str): Turnkey API base URL (defaults from .env or default value).
            api_public_key (str): Turnkey API public key.
            api_private_key (str): Turnkey API private key.
            org_id (str): Turnkey organization ID.

        Raises:
            ValueError: If required configuration parameters are missing.
        """
        # Load .env file
        load_dotenv()

        # Set defaults or read from .env
        self.base_url = base_url or os.getenv("TURNKEY_BASE_URL") or "https://api.turnkey.com"
        self.api_public_key = api_public_key or os.getenv("TURNKEY_API_PUBLIC_KEY")
        self.api_private_key = api_private_key or os.getenv("TURNKEY_API_PRIVATE_KEY")
        self.org_id = org_id or os.getenv("TURNKEY_ORG_ID")

        # Validate configuration
        if not all([self.base_url, self.api_public_key, self.api_private_key, self.org_id]):
            raise ValueError(
                "Missing required Turnkey config: TURNKEY_BASE_URL, TURNKEY_API_PUBLIC_KEY, "
                "TURNKEY_API_PRIVATE_KEY, TURNKEY_ORG_ID"
            )

        # Default payload (for whoami etc.)
        self._payload = {"organizationId": self.org_id}
        self._payload_str = json.dumps(self._payload)

    def _get_x_stamp(self, payload_str=None):
        """
        Generate X-Stamp signature for Turnkey API authentication.

        Args:
            payload_str (str, optional): Payload string to sign, defaults to organization ID payload.

        Returns:
            str: URL-safe Base64 encoded stamp.
        """
        payload_str = payload_str or self._payload_str
        try:
            # Derive private key
            private_key = ec.derive_private_key(int(self.api_private_key, 16), ec.SECP256R1())
            # Create signature
            signature = private_key.sign(payload_str.encode(), ec.ECDSA(hashes.SHA256()))
            # Create stamp
            stamp = {
                "publicKey": self.api_public_key,
                "scheme": "SIGNATURE_SCHEME_TK_API_P256",
                "signature": signature.hex(),
            }
            return urlsafe_b64encode(json.dumps(stamp).encode()).decode().rstrip("=")
        except Exception as e:
            raise ValueError(f"Failed to generate X-Stamp: {e}")

    def _send_post_request(self, endpoint, data=None):
        """
        Send POST request to Turnkey API.

        Args:
            endpoint (str): API endpoint path (e.g., /public/v1/query/whoami).
            data (dict, optional): Request body data, automatically serialized to JSON.

        Returns:
            dict: Parsed JSON response.

        Raises:
            requests.RequestException: If request fails.
        """
        headers = {
            "Content-Type": "application/json",
            "X-Stamp": self._get_x_stamp(json.dumps(data) if data else None),
        }
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                data=json.dumps(data) if data else self._payload_str
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"API request failed: {e}")

    def whoami(self):
        """
        Call whoami API to get organization information.

        Returns:
            dict: JSON response containing organization information.
        """
        endpoint = "/public/v1/query/whoami"
        return self._send_post_request(endpoint)

    def import_private_key(
            self,
            user_id,
            private_key_name,
            encrypted_bundle,
            curve="CURVE_SECP256K1",
            address_formats=["ADDRESS_FORMAT_ETHEREUM"]
    ):
        """
        Import private key to Turnkey.

        Args:
            user_id (str): User ID.
            private_key_name (str): Private key name.
            encrypted_bundle (str): Encrypted private key bundle.
            curve (str): Elliptic curve type, defaults to CURVE_SECP256K1.
            address_formats (list): Address format list, defaults to ["ADDRESS_FORMAT_ETHEREUM"].

        Returns:
            dict: JSON response containing imported private key information.
        """
        endpoint = "/public/v1/submit/import_private_key"
        data = {
            "type": "ACTIVITY_TYPE_IMPORT_PRIVATE_KEY",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "userId": user_id,
                "privateKeyName": private_key_name,
                "encryptedBundle": encrypted_bundle,
                "curve": curve,
                "addressFormats": address_formats
            }
        }
        return self._send_post_request(endpoint, data)

    def sign_evm_transaction(self, sign_with, unsigned_tx):
        """
        Sign EVM transaction using Turnkey.

        Args:
            sign_with (str): Signing identity (wallet account address / private key address / private key ID).
            unsigned_tx (str): Raw unsigned transaction (hex string).

        Returns:
            dict: JSON response containing signing result, see signTransactionResult.signedTransaction.

        Reference:
            https://docs.turnkey.com/api-reference/activities/sign-transaction
        """
        endpoint = "/public/v1/submit/sign_transaction"
        data = {
            "type": "ACTIVITY_TYPE_SIGN_TRANSACTION_V2",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "signWith": sign_with,
                "unsignedTransaction": unsigned_tx,
                "type": "TRANSACTION_TYPE_ETHEREUM"
            }
        }
        return self._send_post_request(endpoint, data)

    def _sign_raw_payload(self, sign_with, payload, encoding, hash_function):
        """
        Internal method: call sign_raw_payload activity.

        Args:
            sign_with (str): Signing identity (wallet account address / private key address / private key ID).
            payload (str): Raw payload to be signed.
            encoding (str): Payload encoding (e.g., PAYLOAD_ENCODING_TEXT_UTF8 / PAYLOAD_ENCODING_EIP712).
            hash_function (str): Hash function (e.g., HASH_FUNCTION_KECCAK256 / HASH_FUNCTION_NOT_APPLICABLE).

        Returns:
            dict: Activity response (result contains r/s/v or other fields).

        Reference:
            https://docs.turnkey.com/api-reference/activities/sign-raw-payload
        """
        endpoint = "/public/v1/submit/sign_raw_payload"
        data = {
            "type": "ACTIVITY_TYPE_SIGN_RAW_PAYLOAD_V2",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "signWith": sign_with,
                "payload": payload,
                "encoding": encoding,
                "hashFunction": hash_function
            }
        }
        return self._send_post_request(endpoint, data)

    def sign_typed_data(self, sign_with, typed_data):
        """
        Sign EIP-712 structured data.

        Args:
            sign_with (str): Signing identity (wallet account address / private key address / private key ID).
            typed_data (dict|str): EIP-712 structure (domain/types/message) or its JSON string.

        Returns:
            dict: Activity response, result contains r/s/v.

        Note:
            - encoding uses PAYLOAD_ENCODING_EIP712
            - hashFunction uses HASH_FUNCTION_NOT_APPLICABLE (server completes EIP-712 spec hashing)
        """
        payload_str = typed_data if isinstance(typed_data, str) else json.dumps(typed_data)
        return self._sign_raw_payload(
            sign_with=sign_with,
            payload=payload_str,
            encoding="PAYLOAD_ENCODING_EIP712",
            hash_function="HASH_FUNCTION_NOT_APPLICABLE",
        )

    def sign_message(self, sign_with, message, use_keccak256=True):
        """
        Sign arbitrary message (defaults to KECCAK256 following Ethereum convention).

        Args:
            sign_with (str): Signing identity (wallet account address / private key address / private key ID).
            message (str|bytes): Text to be signed; bytes will be decoded as UTF-8.
            use_keccak256 (bool): Whether to use KECCAK256 as hash function (default True).

        Returns:
            dict: Activity response, result contains r/s/v.
        """
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        hash_fn = "HASH_FUNCTION_KECCAK256" if use_keccak256 else "HASH_FUNCTION_SHA256"
        return self._sign_raw_payload(
            sign_with=sign_with,
            payload=message,
            encoding="PAYLOAD_ENCODING_TEXT_UTF8",
            hash_function=hash_fn,
        )

    def get_activity(self, activity_id):
        """
        Query Activity details.

        Args:
            activity_id (str): Activity ID.

        Returns:
            dict: Activity details.

        Reference:
            https://docs.turnkey.com/api-reference/queries/get-activity
        """
        endpoint = "/public/v1/query/get_activity"
        data = {
            "organizationId": self.org_id,
            "activityId": activity_id,
        }
        return self._send_post_request(endpoint, data)

    def list_activities(self, limit=None, before=None, after=None, filter_by_status=None, filter_by_type=None):
        """
        List activities within organization (paginated).

        Args:
            limit (str|None): Number per page.
            before (str|None): Pagination cursor (before).
            after (str|None): Pagination cursor (after).
            filter_by_status (list|None): Filter by activity status (e.g., ['ACTIVITY_STATUS_COMPLETED']).
            filter_by_type (list|None): Filter by activity type (e.g., ['ACTIVITY_TYPE_SIGN_TRANSACTION_V2']).

        Returns:
            dict: Activity list.

        Reference:
            https://docs.turnkey.com/api-reference/queries/list-activities
        """
        endpoint = "/public/v1/query/list_activities"
        data = {
            "organizationId": self.org_id,
        }
        
        # Add pagination options
        if any([limit, before, after]):
            data["paginationOptions"] = {}
            if limit:
                data["paginationOptions"]["limit"] = str(limit)
            if before:
                data["paginationOptions"]["before"] = before
            if after:
                data["paginationOptions"]["after"] = after
        
        # Add filters
        if filter_by_status:
            data["filterByStatus"] = filter_by_status
        if filter_by_type:
            data["filterByType"] = filter_by_type
            
        return self._send_post_request(endpoint, data)

    def get_policy_evaluations(self, activity_id):
        """
        Query policy evaluation results for an Activity (if available).

        Args:
            activity_id (str): Activity ID.

        Returns:
            dict: Policy evaluation details.

        Reference:
            https://docs.turnkey.com/api-reference/queries/get-policy-evaluations
        """
        endpoint = "/public/v1/query/get_policy_evaluations"
        data = {
            "organizationId": self.org_id,
            "activityId": activity_id,
        }
        return self._send_post_request(endpoint, data)

    def get_private_key(self, private_key_id):
        """
        Query information for specified private key.

        Args:
            private_key_id (str): Private key ID.

        Returns:
            dict: JSON response containing private key information.
        """
        endpoint = "/public/v1/query/get_private_key"
        data = {
            "organizationId": self.org_id,
            "privateKeyId": private_key_id
        }
        return self._send_post_request(endpoint, data)

    def create_wallet(
            self,
            wallet_name,
            accounts,
            mnemonic_length=24
    ):
        """
        Create new wallet.

        Args:
            wallet_name (str): Wallet name.
            accounts (list): Account configuration list, each account contains curve, pathFormat, path, addressFormat.
            mnemonic_length (int): Mnemonic length (default 24).

        Returns:
            dict: JSON response containing new wallet information.
        """
        endpoint = "/public/v1/submit/create_wallet"
        data = {
            "type": "ACTIVITY_TYPE_CREATE_WALLET",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "walletName": wallet_name,
                "accounts": accounts,
                "mnemonicLength": mnemonic_length
            }
        }
        return self._send_post_request(endpoint, data)

    def create_wallet_accounts(
            self,
            wallet_id,
            accounts
    ):
        """
        Add accounts to existing wallet.

        Args:
            wallet_id (str): Wallet ID.
            accounts (list): New account configuration list, each account contains curve, pathFormat, path, addressFormat.

        Returns:
            dict: JSON response containing new account information.
        """
        endpoint = "/public/v1/submit/create_wallet_accounts"
        data = {
            "type": "ACTIVITY_TYPE_CREATE_WALLET_ACCOUNTS",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "walletId": wallet_id,
                "accounts": accounts
            }
        }
        return self._send_post_request(endpoint, data)

    def get_wallet(self, wallet_id):
        """
        Query information for specified wallet.

        Args:
            wallet_id (str): Wallet ID.

        Returns:
            dict: JSON response containing wallet information.
        """
        endpoint = "/public/v1/query/get_wallet"
        data = {
            "organizationId": self.org_id,
            "walletId": wallet_id
        }
        return self._send_post_request(endpoint, data)

    def get_wallet_account(self, wallet_id, address=None, path=None):
        """
        Query information for specified wallet account.

        Args:
            wallet_id (str): Wallet ID.
            address (str, optional): Account address.
            path (str, optional): Account path (e.g., m/44'/60'/0'/0/0).

        Returns:
            dict: JSON response containing account information.

        Raises:
            ValueError: If neither address nor path is provided.
        """
        if not (address or path):
            raise ValueError("At least one of address or path must be provided")

        endpoint = "/public/v1/query/get_wallet_account"
        data = {
            "organizationId": self.org_id,
            "walletId": wallet_id
        }
        if address:
            data["address"] = address
        if path:
            data["path"] = path
        return self._send_post_request(endpoint, data)

    def list_wallets(self):
        """
        List all wallets in the organization.

        Returns:
            dict: JSON response containing wallet list.
        """
        endpoint = "/public/v1/query/list_wallets"
        data = {
            "organizationId": self.org_id
        }
        return self._send_post_request(endpoint, data)

    def list_wallet_accounts(self, wallet_id, limit=None, before=None, after=None):
        """
        List account list for specified wallet.

        Args:
            wallet_id (str): Wallet ID.
            limit (str, optional): Number of accounts returned per page.
            before (str, optional): Pagination cursor, returns accounts before this ID.
            after (str, optional): Pagination cursor, returns accounts after this ID.

        Returns:
            dict: JSON response containing account list.
        """
        endpoint = "/public/v1/query/list_wallet_accounts"
        data = {
            "organizationId": self.org_id,
            "walletId": wallet_id
        }
        if any([limit, before, after]):
            data["paginationOptions"] = {}
            if limit:
                data["paginationOptions"]["limit"] = str(limit)
            if before:
                data["paginationOptions"]["before"] = before
            if after:
                data["paginationOptions"]["after"] = after
        return self._send_post_request(endpoint, data)

    def init_import_wallet(self, user_id):
        """
        Initialize wallet import process, generate import_bundle.

        Args:
            user_id (str): User ID.

        Returns:
            dict: JSON response containing import_bundle.
        """
        endpoint = "/public/v1/submit/init_import_wallet"
        data = {
            "type": "ACTIVITY_TYPE_INIT_IMPORT_WALLET",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "userId": user_id
            }
        }
        return self._send_post_request(endpoint, data)

    def encrypt_wallet(self, mnemonic, user_id, import_bundle, encryption_key_name="demo-encryption-key"):
        """
        Encrypt mnemonic using Turnkey CLI, generate encrypted_bundle.

        Args:
            mnemonic (str): Mnemonic phrase (12/15/18/21/24 words).
            user_id (str): User ID.
            import_bundle (str): import_bundle obtained from init_import_wallet.
            encryption_key_name (str): Encryption key name, defaults to demo-encryption-key.

        Returns:
            str: Encrypted encrypted_bundle.

        Raises:
            RuntimeError: If CLI command fails or turnkey CLI is not installed.
        """
        try:
            # Ensure turnkey CLI is installed
            subprocess.run(["turnkey", "help"], check=True, capture_output=True)

            # Temporary file paths
            import_bundle_file = "import_bundle.txt"
            encrypted_bundle_file = "encrypted_bundle.txt"

            # Write import_bundle to temporary file
            with open(import_bundle_file, "w") as f:
                f.write(import_bundle)

            # Construct CLI encryption command
            cmd = [
                "turnkey", "encrypt",
                "--user", user_id,
                "--import-bundle-input", import_bundle_file,
                "--plaintext-input", "/dev/fd/3",
                "--encrypted-bundle-output", encrypted_bundle_file,
                "--encryption-key-name", encryption_key_name
            ]

            # Execute CLI command, pass in mnemonic
            process = subprocess.run(
                cmd,
                input=mnemonic.encode(),
                capture_output=True,
                text=True,
                check=True
            )

            # Read encrypted_bundle
            with open(encrypted_bundle_file, "r") as f:
                encrypted_bundle = f.read().strip()

            # Clean up temporary files
            os.remove(import_bundle_file)
            os.remove(encrypted_bundle_file)

            return encrypted_bundle

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to encrypt wallet: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Turnkey CLI not installed. Please install it: https://github.com/tkhq/tkcli")


    def encrypt_private_key(self, private_key, user_id, import_bundle, key_format="hexadecimal",
                            encryption_key_name="demo-encryption-key"):
        """
        Encrypt private key using Turnkey CLI, generate encrypted_bundle, equivalent to:
        `turnkey encrypt --import-bundle-input "./import_bundle.txt" --plaintext-input /dev/fd/3 --key-format "hexadecimal" --encrypted-bundle-output "./encrypted_bundle.txt"`

        Args:
            private_key (str): Private key string (hexadecimal or Solana format).
            user_id (str): User ID.
            import_bundle (str): import_bundle obtained from init_import_private_key.
            key_format (str): Private key format, defaults to "hexadecimal" (supports "hexadecimal", "solana").
            encryption_key_name (str): Encryption key name, defaults to "demo-encryption-key".

        Returns:
            str: Encrypted encrypted_bundle (Base64 encoded string).

        Raises:
            ValueError: If private_key, user_id, import_bundle is empty or key_format is invalid.
            RuntimeError: If CLI command fails or turnkey CLI is not installed.
        """
        if not private_key:
            raise ValueError("private_key is required")
        if not user_id:
            raise ValueError("user_id is required")
        if not import_bundle:
            raise ValueError("import_bundle is required")
        if key_format not in ["hexadecimal", "solana"]:
            raise ValueError("key_format must be 'hexadecimal' or 'solana'")

        try:
            # Verify Turnkey CLI is installed
            subprocess.run(["turnkey", "help"], check=True, capture_output=True)

            # Temporary file paths
            import_bundle_file = "import_bundle.txt"
            encrypted_bundle_file = "encrypted_bundle.txt"

            # Write import_bundle to temporary file
            with open(import_bundle_file, "w") as f:
                f.write(import_bundle)

            # Construct CLI command
            cmd = [
                "turnkey", "encrypt",
                "--user", user_id,
                "--import-bundle-input", import_bundle_file,
                "--plaintext-input", "/dev/fd/3",
                "--key-format", key_format,
                "--encrypted-bundle-output", encrypted_bundle_file,
                "--encryption-key-name", encryption_key_name
            ]

            # Execute CLI command, pass in private key
            process = subprocess.run(
                cmd,
                input=private_key.encode(),
                capture_output=True,
                text=True,
                check=True
            )

            # Read encrypted_bundle
            with open(encrypted_bundle_file, "r") as f:
                encrypted_bundle = f.read().strip()

            # Verify encrypted_bundle is not empty
            if not encrypted_bundle:
                raise RuntimeError("Generated encrypted_bundle is empty")

            # Clean up temporary files
            os.remove(import_bundle_file)
            os.remove(encrypted_bundle_file)

            return encrypted_bundle

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to encrypt private key: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Turnkey CLI not installed. Please install it: https://github.com/tkhq/tkcli")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during encryption: {e}")

    def init_import_private_key(self, user_id):
        """
        Initialize private key import process, generate import_bundle.

        Args:
            user_id (str): User ID.

        Returns:
            dict: JSON response containing import_bundle.
        """
        endpoint = "/public/v1/submit/init_import_private_key"
        data = {
            "type": "ACTIVITY_TYPE_INIT_IMPORT_PRIVATE_KEY",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "userId": user_id
            }
        }
        return self._send_post_request(endpoint, data)

    def import_wallet(self, user_id, wallet_name, encrypted_bundle, accounts=None):
        """
        Import wallet to Turnkey.

        Args:
            user_id (str): User ID.
            wallet_name (str): Wallet name.
            encrypted_bundle (str): Encrypted mnemonic bundle.
            accounts (list, optional): Account configuration list, each account contains curve, pathFormat, path, addressFormat.

        Returns:
            dict: JSON response containing imported wallet information.
        """
        endpoint = "/public/v1/submit/import_wallet"
        data = {
            "type": "ACTIVITY_TYPE_IMPORT_WALLET",
            "timestampMs": str(int(time.time() * 1000)),
            "organizationId": self.org_id,
            "parameters": {
                "userId": user_id,
                "walletName": wallet_name,
                "encryptedBundle": encrypted_bundle,
                "accounts": accounts or [
                    {
                        "curve": "CURVE_SECP256K1",
                        "pathFormat": "PATH_FORMAT_BIP32",
                        "path": "m/44'/60'/0'/0/0",
                        "addressFormat": "ADDRESS_FORMAT_ETHEREUM"
                    }
                ]
            }
        }
        return self._send_post_request(endpoint, data)



import CryptoJS from "crypto-js";

const CryptoHelper = (encryptedUrl: string) => {
    return (password: string) => {
        const bytes = CryptoJS.AES.decrypt(encryptedUrl, password);
        window.location.href = bytes.toString(CryptoJS.enc.Utf8);
    };
}

export default CryptoHelper;

"use client";

import React, {useState} from "react";
import CryptoJS from "crypto-js";
import {Button} from "@/components/ui/button"; // Adjust to your project's Button component path
import {ArrowRight} from "lucide-react"; // Adjust to your project's ArrowRight component path

const CryptoUrl = () => {
    const [url, setUrl] = useState(""); // Input for the encrypted link
    const [password, setPassword] = useState(""); // Input for the password
    const [error, setError] = useState(""); // Error message

    const handleDecrypt = () => {
        try {
            const bytes = CryptoJS.AES.decrypt(url, password);
            window.location.href = bytes.toString(CryptoJS.enc.Utf8);
        } catch {
            setError("Failed to decrypt. Check the password and the encrypted string.");
        }
    };

    const handleEncrypt = () => {
        const encrypted = CryptoJS.AES.encrypt(url, password).toString();
        setUrl(encrypted);
    }

    return (
        <div className="flex flex-col gap-4">
            <input
                type="text"
                placeholder="Enter the link"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="p-2 border rounded-md"
            />

            <input
                type="password"
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="p-2 border rounded-md"
            />

            {error && <p className="text-red-500">{error}</p>}

            {url}

            <Button size="lg" variant="secondary" onClick={handleDecrypt}>
                Get Started <ArrowRight className="ml-2"/>
            </Button>

            <Button size="lg" variant="secondary"
                    onClick={handleEncrypt}>
                Encrypt
            </Button>
        </div>
    );
}

export default CryptoUrl;

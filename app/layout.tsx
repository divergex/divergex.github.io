import type {Metadata} from "next";
import localFont from "next/font/local";
import React from "react";
import "./globals.css";
import DxHeader from "@/app/header";
import DxFooter from "@/app/footer";

const geistSans = localFont({
    src: "./fonts/GeistVF.woff",
    variable: "--font-geist-sans",
    weight: "100 900",
});
const geistMono = localFont({
    src: "./fonts/GeistMonoVF.woff",
    variable: "--font-geist-mono",
    weight: "100 900",
});

export const metadata: Metadata = {
    title: "divergex.io",
    description: "Divergex is a collection of free and open source quantitative frameworks and libraries for algorithmic trading and quantitative research.",
};

export default function RootLayout({
                                       children,
                                   }: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
        <body className={`flex flex-col min-h-screen ${geistSans.variable} ${geistMono.variable} antialiased`}>
        <header className="bg-black z-50 w-full fixed top-0 left-0 ">
            <DxHeader/>
        </header>
        <main className="flex flex-grow pt-[80px] min-h-[calc(100vh-256px)]">
            {children}
        </main>
        <footer className="bg-black py-8">
            <DxFooter/>
        </footer>
        </body>
        </html>
    );
}

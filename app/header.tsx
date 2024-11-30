"use client";

import React, {useState} from 'react';
import Image from 'next/image';
import Link from 'next/link';
import {FaGithub} from 'react-icons/fa';
import {IconContext} from 'react-icons';
import {CloseIcon} from "next/dist/client/components/react-dev-overlay/internal/icons/CloseIcon";

function DxHeader() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
        if (!isMenuOpen) {
            document.body.classList.add('overflow-hidden'); // Disable scrolling
        } else {
            document.body.classList.remove('overflow-hidden'); // Re-enable scrolling
        }
    };

    return (
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
            <div className="w-full md:w-auto flex items-center justify-between">
                <Link
                    className="flex items-center"
                    href={"/"}>
                    <Image src={"/divergex.png"} alt="divergex" width={40} height={40}/>
                    <h1 className="text-2xl ml-2 font-bold bg-gradient-to-r from-white via-gray-400 to-gray-500 bg-clip-text text-transparent">
                        divergex
                    </h1>
                </Link>

                <button
                    className="flex lg:hidden text-white focus:outline-none"
                    onClick={toggleMenu}
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        className="w-6 h-6"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M4 6h16M4 12h16m-7 6h7"
                        />
                    </svg>
                </button>
            </div>
            <nav
                className={`flex md:hidden w-full md:w-auto absolute top-0 left-0 h-screen bg-black flex-col justify-center md:justify-end transform ${
                    isMenuOpen ? 'translate-x-0' : '-translate-x-full'
                } transition-transform duration-300 lg:translate-x-0 lg:h-auto lg:flex-row`}
            >
                <div className="flex py-5 px-3 w-full justify-end">
                    <button
                        className="text-white lg:hidden"
                        onClick={toggleMenu}
                    >
                        <CloseIcon/>
                    </button>
                </div>
                <div
                    className="w-full px-4 flex md:hidden flex-col space-y-6 py-4 items-center lg:flex-row lg:space-x-4 lg:space-y-0">
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-white via-gray-400 to-gray-500 bg-clip-text text-transparent">
                        divergex
                    </h1>
                    <NavItems/>
                </div>
            </nav>
            <nav className={`hidden md:flex w-full justify-end`}>
                <div className="w-full flex space-x-4">
                    <NavItems/>
                </div>
            </nav>
        </div>
    );
}

function NavItems() {
    return (
        <ul className="w-full mt-2 space-y-10 flex flex-col md:flex-row items-center justify-end lg:space-x-4 lg:space-y-0">
            <li className="w-full md:w-fit">
                <Link
                    href={"/#projects"}
                    className="block w-full text-center text-white px-4 py-2 rounded hover:text-black hover:bg-white border border-gray-400"
                >
                    Projects
                </Link>
            </li>
            <li className="w-full md:w-fit">
                <Link
                    href={"/#docs"}
                    className="block w-full text-center text-white px-4 py-2 rounded hover:text-black hover:bg-white border border-gray-400"
                >
                    Documentation
                </Link>
            </li>
            <li className="w-full md:w-fit">
                <Link
                    href={"/about"}
                    className="block w-full text-center text-white px-4 py-2 rounded hover:text-black hover:bg-white border border-gray-400"
                >
                    About
                </Link>
            </li>
            <li className="w-full md:w-fit">
                <Link href="https://github.com/divergex/" className="block w-full">
                    <IconContext.Provider value={{color: "white", size: "2em"}}>
                        <FaGithub className="mx-auto hover:scale-125 transition-transform duration-300 ease-in-out"/>
                    </IconContext.Provider>
                </Link>
            </li>
        </ul>
    )
}

export default DxHeader;
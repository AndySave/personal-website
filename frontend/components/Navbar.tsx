
"use client";

import { useEffect, useState } from "react";

export default function Navbar() {
    const [isScrolled, setIsScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 0);
        };

        window.addEventListener("scroll", handleScroll);
        return () => {
            window.removeEventListener("scroll", handleScroll);
        };
    }, []);  

    return (
        <nav 
            className={`fixed top-0 left-0 w-full flex justify-end gap-10 p-6 text-2xl z-50
                        bg-transparent backdrop-blur-md
                        ${isScrolled ? "shadow-xl/30 border-b border-(--surface-light)" : "shadow-none"}`}
        >
            <a href="/#about">Home</a>
            <a href="/#projects">Projects</a>
            <a href="/#contact">Contact</a>
        </nav>
    )
}

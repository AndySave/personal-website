
import { ReactNode } from "react"

interface Props {
    children: React.ReactNode;
    onClick: () => void;
}


export default function TrainButton( {children, onClick}: Props ) {
    return (
        <button onClick={onClick}>{children}</button>
    )
}
